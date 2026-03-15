import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import json
import tempfile
import re
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# --- Performance Logger ---
class PerfLogger:
    """Tracks execution timing for every section of the app."""

    def __init__(self):
        self.entries = []  # [{name, level, duration_ms, timestamp, status}]
        self._stack = []   # stack of (name, start_time) for nesting
        self._run_start = time.perf_counter()

    def reset(self):
        self.entries = []
        self._stack = []
        self._run_start = time.perf_counter()

    @contextmanager
    def track(self, name, level="INFO"):
        """Context manager to time a block of code."""
        start = time.perf_counter()
        depth = len(self._stack)
        self._stack.append((name, start))
        indent = "  " * depth
        entry = {
            "name": f"{indent}{name}",
            "level": level,
            "duration_ms": 0,
            "timestamp": f"{(start - self._run_start)*1000:.0f}ms",
            "status": "running",
        }
        idx = len(self.entries)
        self.entries.append(entry)
        try:
            yield
            elapsed = (time.perf_counter() - start) * 1000
            self.entries[idx]["duration_ms"] = round(elapsed, 1)
            self.entries[idx]["status"] = "slow" if elapsed > 1000 else "ok"
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self.entries[idx]["duration_ms"] = round(elapsed, 1)
            self.entries[idx]["status"] = "error"
            raise
        finally:
            self._stack.pop()

    def log(self, name, level="INFO"):
        """Log a single event (no timing)."""
        offset = (time.perf_counter() - self._run_start) * 1000
        self.entries.append({
            "name": name,
            "level": level,
            "duration_ms": 0,
            "timestamp": f"{offset:.0f}ms",
            "status": "info",
        })

    def total_ms(self):
        return round((time.perf_counter() - self._run_start) * 1000, 1)


# Global logger — recreated each run
perf = PerfLogger()

# --- Page Config ---
st.set_page_config(
    page_title="Parquet Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 15px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State ---
def init_session():
    defaults = {
        "df": None,
        "parquet_file": None,
        "metadata": None,
        "file_path": None,
        "query_history": [],
        "duckdb_conn": None,
        "browse_path": os.path.expanduser("~"),
        "file_loaded": False,
        "temp_tables": {},  # {name: DataFrame}
        "anthropic_api_key": "",
        "ai_chat_history": [],  # [{role, content}, ...]
        "ai_generated_sql": "",  # last SQL from AI, pending review
        "dev_mode": False,
        "perf_history": deque(maxlen=20),  # last N run logs
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# --- Helper Functions ---
def get_file_size(file_path):
    size = os.path.getsize(file_path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def get_memory_usage(df):
    mem = df.memory_usage(deep=True).sum()
    for unit in ["B", "KB", "MB", "GB"]:
        if mem < 1024:
            return f"{mem:.2f} {unit}"
        mem /= 1024
    return f"{mem:.2f} TB"


@st.cache_data(show_spinner=False)
def _get_memory_usage_cached(file_path):
    """Cached: estimate memory from file size instead of reading the entire file."""
    size = os.path.getsize(file_path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"~{size:.1f} {unit}"
        size /= 1024
    return f"~{size:.1f} TB"


@st.cache_data(show_spinner="Reading parquet file...")
def _read_parquet_to_df(file_path):
    """Cached: read parquet file into pandas DataFrame."""
    pf = pq.ParquetFile(file_path)
    df = pf.read().to_pandas()
    return df


@st.cache_data(show_spinner=False)
def _extract_metadata(file_path):
    """Cached: extract parquet metadata (schema, row groups, etc.)."""
    pf = pq.ParquetFile(file_path)
    metadata = {
        "num_rows": pf.metadata.num_rows,
        "num_columns": pf.metadata.num_columns,
        "num_row_groups": pf.metadata.num_row_groups,
        "format_version": pf.metadata.format_version,
        "created_by": pf.metadata.created_by,
        "serialized_size": pf.metadata.serialized_size,
        "schema": pf.schema_arrow,
        "row_groups": [],
    }

    for i in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(i)
        rg_info = {
            "num_rows": rg.num_rows,
            "total_byte_size": rg.total_byte_size,
            "columns": [],
        }
        for j in range(rg.num_columns):
            col = rg.column(j)
            col_info = {
                "name": col.path_in_schema,
                "compression": str(col.compression),
                "encodings": str(col.encodings),
                "total_compressed_size": col.total_compressed_size,
                "total_uncompressed_size": col.total_uncompressed_size,
                "physical_type": str(col.physical_type),
            }
            if col.is_stats_set:
                col_info["min"] = str(col.statistics.min)
                col_info["max"] = str(col.statistics.max)
                col_info["null_count"] = col.statistics.null_count
                col_info["num_values"] = col.statistics.num_values
                col_info["distinct_count"] = col.statistics.distinct_count
            rg_info["columns"].append(col_info)
        metadata["row_groups"].append(rg_info)

    return metadata


def _build_column_summary(df):
    """Build column summary from the already-loaded DataFrame."""
    summary_data = []
    n = len(df)
    for col_name in df.columns:
        col = df[col_name]
        info = {
            "Column": col_name,
            "Type": str(col.dtype),
            "Non-Null": int(col.notna().sum()),
            "Null": int(col.isna().sum()),
            "Null %": round(col.isna().sum() / n * 100, 2) if n > 0 else 0,
            "Unique": int(col.nunique()),
        }
        if pd.api.types.is_numeric_dtype(col):
            info["Min"] = col.min()
            info["Max"] = col.max()
            info["Mean"] = round(col.mean(), 4) if col.notna().any() else None
        summary_data.append(info)
    return pd.DataFrame(summary_data)


def _compute_null_counts(df):
    """Compute null counts per column from already-loaded DataFrame."""
    null_counts = df.isnull().sum()
    result = pd.DataFrame({
        "Column": null_counts.index,
        "Nulls": null_counts.values,
        "Percentage": (null_counts.values / len(df) * 100).round(2),
    })
    return result[result["Nulls"] > 0].sort_values("Nulls", ascending=False)


def _compute_dtype_counts(df):
    """Compute dtype distribution from already-loaded DataFrame."""
    counts = df.dtypes.astype(str).value_counts().reset_index()
    counts.columns = ["Type", "Count"]
    return counts


@st.cache_data(show_spinner=False)
def _compute_profile_stats(_col_data_hash, col_data, col_name, total_rows):
    """Cached: compute column profiling stats."""
    stats = {
        "Data Type": str(col_data.dtype),
        "Count": f"{col_data.count():,}",
        "Null Count": f"{col_data.isna().sum():,}",
        "Null %": f"{col_data.isna().sum() / total_rows * 100:.2f}%",
        "Unique": f"{col_data.nunique():,}",
        "Unique %": f"{col_data.nunique() / total_rows * 100:.2f}%",
    }
    if pd.api.types.is_numeric_dtype(col_data):
        desc = col_data.describe()
        stats.update({
            "Mean": f"{desc['mean']:.4f}",
            "Std": f"{desc['std']:.4f}",
            "Min": f"{desc['min']:.4f}",
            "25%": f"{desc['25%']:.4f}",
            "Median": f"{desc['50%']:.4f}",
            "75%": f"{desc['75%']:.4f}",
            "Max": f"{desc['max']:.4f}",
            "Skewness": f"{col_data.skew():.4f}",
            "Kurtosis": f"{col_data.kurtosis():.4f}",
            "Zeros": f"{(col_data == 0).sum():,}",
            "Negatives": f"{(col_data < 0).sum():,}",
        })
    if pd.api.types.is_string_dtype(col_data):
        lengths = col_data.dropna().str.len()
        if len(lengths) > 0:
            stats.update({
                "Min Length": int(lengths.min()),
                "Max Length": int(lengths.max()),
                "Mean Length": f"{lengths.mean():.1f}",
                "Empty Strings": int((col_data == "").sum()),
            })
    return stats


def _sync_temp_tables_to_duckdb(conn):
    """Register all session temp tables into the DuckDB connection."""
    for tname, tdf in st.session_state.temp_tables.items():
        try:
            conn.register(tname, tdf)
        except Exception:
            pass


def _build_schema_context(main_df, temp_tables):
    """Build a concise schema description for the AI prompt."""
    lines = []
    lines.append("Table: parquet_data")
    lines.append(f"  Rows: {len(main_df):,}")
    for c in main_df.columns:
        sample_vals = ""
        if pd.api.types.is_string_dtype(main_df[c]) or pd.api.types.is_categorical_dtype(main_df[c]):
            uniques = main_df[c].dropna().unique()[:5].tolist()
            sample_vals = f"  samples: {uniques}"
        lines.append(f"  - {c} ({main_df[c].dtype}){sample_vals}")

    for tname, tdf in temp_tables.items():
        lines.append(f"\nTable: {tname}")
        lines.append(f"  Rows: {len(tdf):,}")
        for c in tdf.columns:
            lines.append(f"  - {c} ({tdf[c].dtype})")

    return "\n".join(lines)


def _call_claude_for_sql(api_key, schema_context, chat_history, user_message):
    """Call Claude API to generate a SQL query based on the user request."""
    if not HAS_ANTHROPIC:
        return None, "The `anthropic` package is not installed. Run: pip install anthropic"

    system_prompt = f"""You are a SQL assistant for DuckDB. The user has loaded data and wants you to write SQL queries.

Available tables and their schemas:
{schema_context}

Rules:
- Write valid DuckDB SQL
- Only write SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
- Always quote column names with double quotes if they contain spaces or special characters
- Return ONLY the SQL query inside a ```sql code block, followed by a brief explanation
- If the user request is unclear, ask for clarification
- Keep queries efficient"""

    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text, None
    except anthropic.AuthenticationError:
        return None, "Invalid API key. Please check your Anthropic API key."
    except Exception as e:
        return None, f"API error: {str(e)}"


def _extract_sql_from_response(text):
    """Extract SQL from a markdown code block in the AI response."""
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: try generic code block
    pattern = r"```\s*(SELECT.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _render_ai_sql_assistant(widget_prefix, target_key):
    """Render the AI SQL assistant chat UI. Sets st.session_state[target_key] when user accepts a query."""
    if not HAS_ANTHROPIC:
        st.warning("Install the `anthropic` package to use the AI assistant: `pip install anthropic`")
        return

    # API key input (in sidebar on first use)
    if not st.session_state.anthropic_api_key:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            key=f"{widget_prefix}_api_key_input",
            help="Your key is stored in session only, never written to disk.",
        )
        if api_key:
            st.session_state.anthropic_api_key = api_key
            st.rerun()
        else:
            st.info("Enter your Anthropic API key above to start chatting.")
            return

    # Chat UI
    st.markdown("**Chat with Claude to generate SQL**")

    # Display chat history
    for idx, msg in enumerate(st.session_state.ai_chat_history):
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

            # If this assistant message contains SQL, show copy button
            extracted = _extract_sql_from_response(msg["content"])
            if extracted:
                col_use, col_copy = st.columns([1, 1])
                with col_use:
                    if st.button(
                        "Use this query",
                        key=f"{widget_prefix}_use_{idx}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state[target_key] = extracted
                        st.rerun()
                with col_copy:
                    st.code(extracted, language="sql")

    # Chat input — use text_input + button since chat_input can only appear once per app
    ai_col1, ai_col2 = st.columns([5, 1])
    with ai_col1:
        user_input = st.text_input(
            "Message",
            placeholder="Describe what you want to query...",
            key=f"{widget_prefix}_chat_input",
            label_visibility="collapsed",
        )
    with ai_col2:
        send_clicked = st.button("Send", key=f"{widget_prefix}_send", type="primary", use_container_width=True)

    if send_clicked and user_input:
        # Add user message to history
        st.session_state.ai_chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Build schema context
        main_df = st.session_state.df
        schema_ctx = _build_schema_context(main_df, st.session_state.temp_tables)

        # Call Claude
        with st.spinner("Claude is thinking..."):
            response_text, error = _call_claude_for_sql(
                st.session_state.anthropic_api_key,
                schema_ctx,
                st.session_state.ai_chat_history[:-1],  # exclude the just-added msg
                user_input,
            )

        if error:
            st.error(error)
            # Remove the user message we just added since it failed
            st.session_state.ai_chat_history.pop()
        else:
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response_text})
            st.rerun()

    # Clear chat button
    if st.session_state.ai_chat_history:
        if st.button("Clear chat", key=f"{widget_prefix}_clear_chat"):
            st.session_state.ai_chat_history = []
            st.rerun()


def load_parquet(source, is_upload=False):
    """Load parquet file from upload or file path."""
    try:
        with perf.track("load_parquet"):
            if is_upload:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                tmp.write(source.getvalue())
                tmp.close()
                file_path = tmp.name
            else:
                file_path = source

            with perf.track("read_parquet_to_df"):
                df = _read_parquet_to_df(file_path)
            perf.log(f"  loaded {len(df):,} rows x {len(df.columns)} cols")

            with perf.track("extract_metadata"):
                metadata = _extract_metadata(file_path)

            st.session_state.df = df
            st.session_state.metadata = metadata
            st.session_state.file_path = file_path

            with perf.track("open_parquet_file"):
                pf = pq.ParquetFile(file_path)
            st.session_state.parquet_file = pf

            with perf.track("register_duckdb"):
                conn = duckdb.connect()
                conn.register("parquet_data", df)
                _sync_temp_tables_to_duckdb(conn)
                st.session_state.duckdb_conn = conn

        return True, "File loaded successfully!"
    except Exception as e:
        return False, f"Error: {str(e)}"


# --- Sidebar ---
with st.sidebar:
    st.markdown('<p class="main-header">Parquet Explorer</p>', unsafe_allow_html=True)
    st.markdown("---")

    load_method = st.radio("Load Method", ["Upload File", "File Path", "Browse Folders"])

    if load_method == "Upload File":
        uploaded = st.file_uploader("Upload Parquet File", type=["parquet"])
        if uploaded and st.button("Load File", type="primary", use_container_width=True):
            with st.spinner("Loading..."):
                ok, msg = load_parquet(uploaded, is_upload=True)
            if ok:
                st.success(msg)
                st.session_state.file_loaded = True
                st.rerun()
            else:
                st.error(msg)

    elif load_method == "File Path":
        file_path = st.text_input("Enter file path", placeholder="/path/to/file.parquet")
        if file_path and st.button("Load File", type="primary", use_container_width=True):
            if os.path.exists(file_path):
                with st.spinner("Loading..."):
                    ok, msg = load_parquet(file_path)
                if ok:
                    st.success(msg)
                    st.session_state.file_loaded = True
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("File not found!")

    elif load_method == "Browse Folders":
        # Interactive folder browser
        current_path = st.session_state.browse_path

        st.text_input("Current path", value=current_path, key="path_input",
                       on_change=lambda: st.session_state.update(
                           browse_path=st.session_state.path_input
                       ))

        if os.path.isdir(current_path):
            # Navigation buttons
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                if st.button("Parent Folder", use_container_width=True):
                    parent = str(Path(current_path).parent)
                    st.session_state.browse_path = parent
                    st.rerun()
            with col_nav2:
                if st.button("Home", use_container_width=True):
                    st.session_state.browse_path = os.path.expanduser("~")
                    st.rerun()

            # List directories and parquet files
            try:
                entries = sorted(os.listdir(current_path))
            except PermissionError:
                entries = []
                st.error("Permission denied!")

            dirs = [e for e in entries if os.path.isdir(os.path.join(current_path, e)) and not e.startswith('.')]
            pq_files = [e for e in entries if e.endswith('.parquet')]

            # Show subdirectories
            if dirs:
                st.markdown("**Folders:**")
                selected_dir = st.selectbox(
                    "Navigate to folder",
                    ["-- select --"] + dirs,
                    key="folder_select",
                    label_visibility="collapsed",
                )
                if selected_dir != "-- select --":
                    if st.button("Open Folder", use_container_width=True):
                        st.session_state.browse_path = os.path.join(current_path, selected_dir)
                        st.rerun()

            # Show parquet files
            if pq_files:
                st.markdown(f"**Parquet Files ({len(pq_files)}):**")
                selected_file = st.selectbox("Select file", pq_files, key="pq_file_select")
                if st.button("Load Selected File", type="primary", use_container_width=True):
                    full_path = os.path.join(current_path, selected_file)
                    with st.spinner("Loading..."):
                        ok, msg = load_parquet(full_path)
                    if ok:
                        st.success(msg)
                        st.session_state.file_loaded = True
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.info("No .parquet files in this folder.")

            # Recursive search option
            if st.checkbox("Search subfolders"):
                with st.spinner("Scanning..."):
                    found = sorted(Path(current_path).rglob("*.parquet"))
                if found:
                    st.markdown(f"**Found {len(found)} files:**")
                    selected_recursive = st.selectbox(
                        "Select file",
                        found,
                        format_func=lambda x: str(x.relative_to(current_path)),
                        key="recursive_pq_select",
                    )
                    if st.button("Load File", type="primary", use_container_width=True, key="load_recursive"):
                        with st.spinner("Loading..."):
                            ok, msg = load_parquet(str(selected_recursive))
                        if ok:
                            st.success(msg)
                            st.session_state.file_loaded = True
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    st.warning("No .parquet files in subfolders.")
        else:
            st.error("Invalid directory path.")

    st.markdown("---")
    if st.session_state.df is not None:
        st.markdown("**Loaded File Info**")
        meta = st.session_state.metadata
        st.markdown(f"- Rows: `{meta['num_rows']:,}`")
        st.markdown(f"- Columns: `{meta['num_columns']}`")
        st.markdown(f"- Row Groups: `{meta['num_row_groups']}`")
        st.markdown(f"- Memory: `{_get_memory_usage_cached(st.session_state.file_path) if st.session_state.file_path else 'N/A'}`")

        if st.session_state.temp_tables:
            st.markdown("---")
            st.markdown(f"**Temp Tables ({len(st.session_state.temp_tables)})**")
            for tname, tdf in st.session_state.temp_tables.items():
                st.markdown(f"- `{tname}` ({len(tdf):,} rows)")

    # --- Developer Mode toggle ---
    st.markdown("---")
    st.session_state.dev_mode = st.toggle(
        "Developer Mode", value=st.session_state.get("dev_mode", False), key="dev_mode_toggle"
    )

# --- Main Content ---
if st.session_state.df is None:
    st.markdown("## Welcome to Parquet Explorer")
    st.markdown("Load a parquet file using the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Features")
        st.markdown("""
        - Schema & metadata inspection
        - Data preview with sorting/filtering
        - SQL queries via DuckDB
        - Column statistics & profiling
        - Interactive visualizations
        - Export to CSV/JSON/Excel
        """)
    with col2:
        st.markdown("### Supported")
        st.markdown("""
        - Upload or browse files
        - Large file handling
        - All parquet data types
        - Nested/complex schemas
        - Multiple row groups
        - Compressed files
        """)
    with col3:
        st.markdown("### Query Engine")
        st.markdown("""
        - Full SQL support (DuckDB)
        - Query history
        - Result export
        - Auto-complete table name
        - Aggregations & joins
        - Window functions
        """)
else:
    perf.log("=== Script rerun started ===")
    df = st.session_state.df
    metadata = st.session_state.metadata

    # --- Always-visible file info and data snapshot ---
    with perf.track("header_metrics"):
        file_display = st.session_state.file_path or "Uploaded file"
        st.markdown(f"### Loaded: `{os.path.basename(file_display)}`")

        info_cols = st.columns(6)
        info_cols[0].metric("Rows", f"{metadata['num_rows']:,}")
        info_cols[1].metric("Columns", metadata['num_columns'])
        info_cols[2].metric("Row Groups", metadata['num_row_groups'])
        info_cols[3].metric("Memory", _get_memory_usage_cached(st.session_state.file_path) if st.session_state.file_path else "N/A")
        if st.session_state.file_path and os.path.exists(st.session_state.file_path):
            info_cols[4].metric("File Size", get_file_size(st.session_state.file_path))
        info_cols[5].metric("Dtypes", df.dtypes.nunique())

    with perf.track("quick_data_view"):
        with st.expander("Quick Data View (first 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True, hide_index=False)

    st.markdown("---")

    # --- Tabs ---
    tab_names = [
        "Overview",
        "Data Preview",
        "Schema",
        "SQL Query",
        "Temp Tables",
        "Column Profiler",
        "Visualize",
        "Row Groups",
        "Export",
    ]
    if st.session_state.dev_mode:
        tab_names.append("Dev Logs")

    tabs = st.tabs(tab_names)

    # ==================== OVERVIEW ====================
    with tabs[0]:
        perf.log("TAB: Overview — start")
        _tab_t = time.perf_counter()
        st.markdown("### Dataset Overview")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{metadata['num_rows']:,}")
        c2.metric("Columns", metadata['num_columns'])
        c3.metric("Row Groups", metadata['num_row_groups'])
        c4.metric("Memory", _get_memory_usage_cached(st.session_state.file_path) if st.session_state.file_path else "N/A")
        if st.session_state.file_path and os.path.exists(st.session_state.file_path):
            c5.metric("File Size", get_file_size(st.session_state.file_path))

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Data Types Distribution")
            with perf.track("overview: dtype_counts"):
                dtype_counts = _compute_dtype_counts(df)
            if "Type" not in dtype_counts.columns:
                dtype_counts.columns = ["Type", "Count"]
            fig = px.pie(dtype_counts, values="Count", names="Type", hole=0.4)
            fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Missing Values")
            # Use cached result to avoid recomputing on every rerun
            if "_null_counts_cache" not in st.session_state:
                with perf.track("overview: null_counts"):
                    st.session_state["_null_counts_cache"] = _compute_null_counts(df)
            null_df = st.session_state["_null_counts_cache"]
            if len(null_df) > 0:
                fig = px.bar(null_df.head(20), x="Column", y="Percentage",
                             text="Nulls", color="Percentage",
                             color_continuous_scale="Reds")
                fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")

        st.markdown("#### Column Summary")
        if st.button("Compute Column Summary", key="compute_col_summary"):
            with perf.track("overview: column_summary"):
                summary_df = _build_column_summary(df)
            st.session_state["_col_summary_cache"] = summary_df
        if "_col_summary_cache" in st.session_state:
            st.dataframe(st.session_state["_col_summary_cache"], use_container_width=True, hide_index=True)
        else:
            st.info("Click above to compute column summary (may take time for large datasets).")

    # ==================== DATA PREVIEW ====================
    with tabs[1]:
        perf.log("TAB: Data Preview — start")
        _tab_t = time.perf_counter()
        st.markdown("### Data Preview")

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            view_mode = st.selectbox("View", ["Head", "Tail", "Sample", "Full"])
        with col2:
            n_rows = st.number_input("Rows to show", min_value=5, max_value=10000, value=100)
        with col3:
            with perf.track("data_preview: column_select"):
                all_col_names = df.columns.tolist()
                use_all = st.checkbox("All columns", value=True, key="dp_all_cols")
                if use_all:
                    selected_cols = all_col_names
                else:
                    selected_cols = st.multiselect("Columns", all_col_names, default=all_col_names[:5], key="dp_cols")

        # Filter section
        with st.expander("Filters", expanded=False):
            filter_col = st.selectbox("Filter column", ["None"] + df.columns.tolist(), key="filter_col")
            filtered_df = df[selected_cols]

            if filter_col != "None":
                col_type = df[filter_col].dtype

                if pd.api.types.is_numeric_dtype(col_type):
                    min_val = float(df[filter_col].min())
                    max_val = float(df[filter_col].max())
                    range_vals = st.slider("Range", min_val, max_val, (min_val, max_val), key="num_filter")
                    filtered_df = filtered_df[
                        (df[filter_col] >= range_vals[0]) & (df[filter_col] <= range_vals[1])
                    ]
                elif pd.api.types.is_string_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
                    unique_vals = df[filter_col].dropna().unique().tolist()
                    if len(unique_vals) <= 100:
                        selected_vals = st.multiselect("Values", unique_vals, key="cat_filter")
                        if selected_vals:
                            filtered_df = filtered_df[df[filter_col].isin(selected_vals)]
                    else:
                        text_filter = st.text_input("Contains", key="text_filter")
                        if text_filter:
                            filtered_df = filtered_df[
                                df[filter_col].astype(str).str.contains(text_filter, case=False, na=False)
                            ]
                elif pd.api.types.is_datetime64_any_dtype(col_type):
                    min_date = df[filter_col].min()
                    max_date = df[filter_col].max()
                    date_range = st.date_input("Date range", [min_date, max_date], key="date_filter")
                    if len(date_range) == 2:
                        filtered_df = filtered_df[
                            (df[filter_col] >= pd.Timestamp(date_range[0])) &
                            (df[filter_col] <= pd.Timestamp(date_range[1]))
                        ]

            st.info(f"Showing {len(filtered_df):,} rows after filter (from {len(df):,} total)")

        # Sort section
        with st.expander("Sort", expanded=False):
            sort_col = st.selectbox("Sort by", ["None"] + selected_cols, key="sort_col")
            sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key="sort_order")
            if sort_col != "None":
                filtered_df = filtered_df.sort_values(sort_col, ascending=(sort_order == "Ascending"))

        # Display
        if view_mode == "Head":
            display_df = filtered_df.head(n_rows)
        elif view_mode == "Tail":
            display_df = filtered_df.tail(n_rows)
        elif view_mode == "Sample":
            display_df = filtered_df.sample(min(n_rows, len(filtered_df)))
        else:
            display_df = filtered_df.head(n_rows)

        with perf.track(f"data_preview: render_dataframe ({len(display_df):,} rows)"):
            st.dataframe(display_df, use_container_width=True, hide_index=False, height=600)

        # Search — uses DuckDB for speed instead of row-by-row pandas apply
        st.markdown("#### Search")
        search_term = st.text_input("Search across all columns", key="global_search")
        if search_term:
            conn = st.session_state.duckdb_conn
            # Build a WHERE clause that checks all columns via CAST to VARCHAR
            safe_term = search_term.replace("'", "''")
            conditions = " OR ".join(
                f'CAST("{c}" AS VARCHAR) ILIKE \'%{safe_term}%\'' for c in df.columns
            )
            try:
                results = conn.execute(
                    f"SELECT * FROM parquet_data WHERE {conditions} LIMIT 500"
                ).fetchdf()
                st.info(f"Showing up to 500 matching rows")
                st.dataframe(results, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Search error: {e}")

    # ==================== SCHEMA ====================
    with tabs[2]:
        perf.log("TAB: Schema — start")
        _tab_t = time.perf_counter()
        st.markdown("### Schema & Metadata")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Arrow Schema")
            schema = metadata["schema"]
            schema_data = []
            for i in range(len(schema)):
                field = schema.field(i)
                schema_data.append({
                    "Index": i,
                    "Name": field.name,
                    "Arrow Type": str(field.type),
                    "Nullable": field.nullable,
                    "Pandas Type": str(df[field.name].dtype),
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### File Metadata")
            meta_info = {
                "Format Version": metadata["format_version"],
                "Created By": metadata["created_by"] or "Unknown",
                "Number of Rows": f"{metadata['num_rows']:,}",
                "Number of Columns": metadata["num_columns"],
                "Number of Row Groups": metadata["num_row_groups"],
                "Serialized Size": f"{metadata['serialized_size']:,} bytes",
            }
            for k, v in meta_info.items():
                st.markdown(f"**{k}:** `{v}`")

            # Key-value metadata
            pf = st.session_state.parquet_file
            if pf.schema_arrow.metadata:
                st.markdown("#### Custom Metadata")
                for k, v in pf.schema_arrow.metadata.items():
                    with st.expander(k.decode() if isinstance(k, bytes) else str(k)):
                        val = v.decode() if isinstance(v, bytes) else str(v)
                        try:
                            parsed = json.loads(val)
                            st.json(parsed)
                        except (json.JSONDecodeError, ValueError):
                            st.code(val)

    # ==================== SQL QUERY ====================
    with tabs[3]:
        perf.log("TAB: SQL Query — start")
        _tab_t = time.perf_counter()
        st.markdown("### SQL Query Engine (DuckDB)")
        tt_names = list(st.session_state.temp_tables.keys())
        tables_hint = "`parquet_data`"
        if tt_names:
            tables_hint += ", " + ", ".join(f"`{t}`" for t in tt_names)
        st.info(f"Available tables: {tables_hint}  |  Example: `SELECT * FROM parquet_data LIMIT 10`")

        # ---- Toggle: Manual vs AI Assistant ----
        sql_mode = st.toggle("Use AI Assistant to write SQL", value=False, key="sql_ai_toggle")

        if sql_mode:
            # AI generates SQL, user reviews and copies to query box
            _render_ai_sql_assistant("sql_tab", "sql_query")
            st.markdown("---")

        # The query box — populated by AI "Use this query" or typed manually
        query = st.text_area(
            "SQL Query",
            value=st.session_state.get("sql_query", "SELECT * FROM parquet_data LIMIT 100"),
            height=120,
            key="sql_query_editor",
        )

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            run_query = st.button("Run Query", type="primary", use_container_width=True)
        with col2:
            clear_history = st.button("Clear History", use_container_width=True)

        if clear_history:
            st.session_state.query_history = []

        if run_query and query.strip():
            try:
                conn = st.session_state.duckdb_conn
                _sync_temp_tables_to_duckdb(conn)
                start = datetime.now()
                result = conn.execute(query).fetchdf()
                elapsed = (datetime.now() - start).total_seconds()

                st.session_state.query_history.append({
                    "query": query,
                    "rows": len(result),
                    "time": f"{elapsed:.3f}s",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })

                st.success(f"Returned {len(result):,} rows in {elapsed:.3f}s")
                st.dataframe(result, use_container_width=True, hide_index=True, height=500)

                # Download query results
                csv_data = result.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    csv_data,
                    "query_results.csv",
                    "text/csv",
                    key="download_query",
                )
            except Exception as e:
                st.error(f"Query Error: {str(e)}")

        # Query templates
        with st.expander("Query Templates"):
            templates = {
                "Row Count": "SELECT COUNT(*) as total_rows FROM parquet_data",
                "Column Stats": "SELECT column_name, COUNT(*) FROM (SELECT UNNEST(columns) FROM parquet_metadata('{path}')) GROUP BY column_name" if st.session_state.file_path else "",
                "Top N Values": f"SELECT \"{df.columns[0]}\", COUNT(*) as cnt FROM parquet_data GROUP BY \"{df.columns[0]}\" ORDER BY cnt DESC LIMIT 10",
                "Distinct Counts": "SELECT " + ", ".join([f'COUNT(DISTINCT "{c}") as "{c}"' for c in df.columns[:10]]) + " FROM parquet_data",
                "Null Counts": "SELECT " + ", ".join([f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) as "{c}_nulls"' for c in df.columns[:10]]) + " FROM parquet_data",
                "Describe Numeric": "SELECT " + ", ".join([f'AVG("{c}") as "avg_{c}", STDDEV("{c}") as "std_{c}"' for c in df.select_dtypes("number").columns[:5]]) + " FROM parquet_data" if len(df.select_dtypes("number").columns) > 0 else "-- No numeric columns",
                "Sample Random": "SELECT * FROM parquet_data USING SAMPLE 10",
            }
            for name, tmpl in templates.items():
                if tmpl:
                    st.code(tmpl, language="sql")

        # Query history
        if st.session_state.query_history:
            with st.expander("Query History"):
                for i, h in enumerate(reversed(st.session_state.query_history)):
                    st.markdown(f"**[{h['timestamp']}]** ({h['time']}, {h['rows']} rows)")
                    st.code(h["query"], language="sql")

    # ==================== TEMP TABLES ====================
    with tabs[4]:
        perf.log("TAB: Temp Tables — start")
        _tab_t = time.perf_counter()
        st.markdown("### Temp Tables")
        st.caption(
            "Create filtered snapshots of your data as temporary tables. "
            "Use them in SQL queries and visualizations."
        )

        temp_tables = st.session_state.temp_tables
        conn = st.session_state.duckdb_conn

        # ---- Create new temp table ----
        st.markdown("---")
        st.markdown("#### Create Temp Table")
        create_method = st.radio(
            "Create from",
            ["SQL Query", "Current Filters (Data Preview)"],
            horizontal=True,
            key="tt_create_method",
        )

        tt_name = st.text_input(
            "Table name",
            placeholder="e.g. filtered_sales, top_customers",
            key="tt_name_input",
        )

        if create_method == "SQL Query":
            tt_ai_mode = st.toggle("Use AI Assistant", value=False, key="tt_ai_toggle")
            if tt_ai_mode:
                _render_ai_sql_assistant("tt_tab", "tt_query")
                st.markdown("---")

            tt_query = st.text_area(
                "SQL Query",
                value=st.session_state.get("tt_query", "SELECT * FROM parquet_data WHERE 1=1 LIMIT 1000"),
                height=100,
                key="tt_query_editor",
            )
            if st.button("Create Table", type="primary", key="tt_create_sql"):
                name = tt_name.strip()
                if not name:
                    st.error("Enter a table name.")
                elif not name.isidentifier():
                    st.error("Table name must be a valid identifier (letters, digits, underscores).")
                elif name == "parquet_data":
                    st.error("Cannot overwrite the main table.")
                else:
                    try:
                        # Sync existing temp tables so the query can reference them
                        _sync_temp_tables_to_duckdb(conn)
                        result_df = conn.execute(tt_query).fetchdf()
                        st.session_state.temp_tables[name] = result_df
                        conn.register(name, result_df)
                        st.success(
                            f"Created **{name}** ({len(result_df):,} rows x {len(result_df.columns)} cols)"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Query error: {e}")

        else:  # From current filters
            st.info(
                "This will save the currently filtered view from the **Data Preview** tab. "
                "Go to Data Preview first to set your filters, then come back here."
            )
            if st.button("Create Table from Filters", type="primary", key="tt_create_filter"):
                name = tt_name.strip()
                if not name:
                    st.error("Enter a table name.")
                elif not name.isidentifier():
                    st.error("Table name must be a valid identifier (letters, digits, underscores).")
                elif name == "parquet_data":
                    st.error("Cannot overwrite the main table.")
                else:
                    # Reconstruct filtered_df from Data Preview state
                    sel_cols = st.session_state.get("columns_multiselect", df.columns.tolist())
                    if not sel_cols:
                        sel_cols = df.columns.tolist()
                    filt_df = df[sel_cols].copy()

                    f_col = st.session_state.get("filter_col", "None")
                    if f_col and f_col != "None" and f_col in df.columns:
                        col_type = df[f_col].dtype
                        if pd.api.types.is_numeric_dtype(col_type):
                            r = st.session_state.get("num_filter")
                            if r:
                                filt_df = filt_df[
                                    (df[f_col] >= r[0]) & (df[f_col] <= r[1])
                                ]
                        elif pd.api.types.is_string_dtype(col_type):
                            vals = st.session_state.get("cat_filter")
                            txt = st.session_state.get("text_filter")
                            if vals:
                                filt_df = filt_df[df[f_col].isin(vals)]
                            elif txt:
                                filt_df = filt_df[
                                    df[f_col].astype(str).str.contains(txt, case=False, na=False)
                                ]

                    st.session_state.temp_tables[name] = filt_df
                    conn.register(name, filt_df)
                    st.success(
                        f"Created **{name}** ({len(filt_df):,} rows x {len(filt_df.columns)} cols)"
                    )
                    st.rerun()

        # ---- List existing temp tables ----
        st.markdown("---")
        st.markdown("#### Existing Temp Tables")

        if not temp_tables:
            st.info("No temp tables yet. Create one above.")
        else:
            # Lightweight summary — no expensive memory computation
            tt_summary = []
            for tname, tdf in temp_tables.items():
                tt_summary.append({
                    "Table": tname,
                    "Rows": f"{len(tdf):,}",
                    "Columns": len(tdf.columns),
                    "Column Names": ", ".join(tdf.columns[:8]) + ("..." if len(tdf.columns) > 8 else ""),
                })
            st.dataframe(pd.DataFrame(tt_summary), use_container_width=True, hide_index=True)

            # ---- Inspect / Modify / Delete a temp table ----
            st.markdown("---")
            selected_tt = st.selectbox(
                "Select table to inspect",
                list(temp_tables.keys()),
                key="tt_inspect_select",
            )
            tt_df = temp_tables[selected_tt]

            inspect_action = st.radio(
                "Action",
                ["Preview", "Schema", "Modify (SQL)", "Rename", "Delete"],
                horizontal=True,
                key="tt_action",
            )

            if inspect_action == "Preview":
                st.markdown(f"**{selected_tt}** — {len(tt_df):,} rows x {len(tt_df.columns)} cols")
                preview_n = st.slider("Rows to show", 10, min(1000, len(tt_df)), 50, key="tt_preview_n")
                st.dataframe(tt_df.head(preview_n), use_container_width=True, hide_index=True)

            elif inspect_action == "Schema":
                # Only compute expensive stats when Schema is explicitly selected
                schema_rows = []
                for c in tt_df.columns:
                    schema_rows.append({
                        "Column": c,
                        "Type": str(tt_df[c].dtype),
                        "Non-Null": int(tt_df[c].notna().sum()),
                        "Null": int(tt_df[c].isna().sum()),
                        "Unique": int(tt_df[c].nunique()),
                    })
                st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

            elif inspect_action == "Modify (SQL)":
                st.markdown(
                    f"Write a SELECT query to **replace** the contents of `{selected_tt}`. "
                    f"You can reference `{selected_tt}` itself, `parquet_data`, or other temp tables."
                )
                modify_query = st.text_area(
                    "Modify query",
                    value=f"SELECT * FROM {selected_tt} WHERE 1=1",
                    height=100,
                    key="tt_modify_query",
                )
                if st.button("Apply Modification", type="primary", key="tt_modify_apply"):
                    try:
                        _sync_temp_tables_to_duckdb(conn)
                        new_df = conn.execute(modify_query).fetchdf()
                        st.session_state.temp_tables[selected_tt] = new_df
                        conn.register(selected_tt, new_df)
                        st.success(
                            f"Updated **{selected_tt}** ({len(new_df):,} rows x {len(new_df.columns)} cols)"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Query error: {e}")

            elif inspect_action == "Rename":
                new_name = st.text_input("New name", value=selected_tt, key="tt_rename_input")
                if st.button("Rename", key="tt_rename_btn"):
                    new_name = new_name.strip()
                    if not new_name:
                        st.error("Enter a name.")
                    elif not new_name.isidentifier():
                        st.error("Must be a valid identifier.")
                    elif new_name == "parquet_data":
                        st.error("Cannot use the main table name.")
                    elif new_name in temp_tables and new_name != selected_tt:
                        st.error(f"Table '{new_name}' already exists.")
                    elif new_name != selected_tt:
                        st.session_state.temp_tables[new_name] = st.session_state.temp_tables.pop(selected_tt)
                        conn.register(new_name, st.session_state.temp_tables[new_name])
                        try:
                            conn.unregister(selected_tt)
                        except Exception:
                            pass
                        st.success(f"Renamed **{selected_tt}** to **{new_name}**")
                        st.rerun()

            elif inspect_action == "Delete":
                st.warning(f"This will permanently delete temp table **{selected_tt}**.")
                if st.button("Confirm Delete", type="primary", key="tt_delete_btn"):
                    del st.session_state.temp_tables[selected_tt]
                    try:
                        conn.unregister(selected_tt)
                    except Exception:
                        pass
                    st.success(f"Deleted **{selected_tt}**")
                    st.rerun()

            # ---- Quick SQL on temp tables ----
            st.markdown("---")
            st.markdown("#### Quick Query on Temp Tables")
            st.caption("All temp tables are available as table names in SQL queries.")
            available = ", ".join(f"`{t}`" for t in temp_tables.keys())
            st.markdown(f"Available: {available}")

    # ==================== COLUMN PROFILER ====================
    with tabs[5]:
        perf.log("TAB: Column Profiler — start")
        _tab_t = time.perf_counter()
        st.markdown("### Column Profiler")

        profile_col = st.selectbox("Select Column", df.columns.tolist(), key="profile_col")

        if st.button("Profile Column", type="primary", key="run_profile"):
            st.session_state["profiler_active"] = profile_col

        if st.session_state.get("profiler_active") != profile_col:
            st.info("Select a column and click **Profile Column** to compute stats.")
        else:
            col_data = df[profile_col]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Basic Stats")
                cache_key = f"{st.session_state.file_path}_{profile_col}"
                stats = _compute_profile_stats(cache_key, col_data, profile_col, len(df))

                for k, v in stats.items():
                    st.markdown(f"**{k}:** `{v}`")

            with col2:
                st.markdown("#### Distribution")
                if pd.api.types.is_numeric_dtype(col_data):
                    fig = px.histogram(df, x=profile_col, nbins=50, marginal="box")
                    fig.update_layout(height=400, margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    fig = px.histogram(df, x=profile_col, nbins=50)
                    fig.update_layout(height=400, margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    top_vals = col_data.value_counts().head(20)
                    fig = px.bar(x=top_vals.index.astype(str), y=top_vals.values,
                                 labels={"x": profile_col, "y": "Count"})
                    fig.update_layout(height=400, margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            # Frequent and rare values
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Most Frequent Values")
                top = col_data.value_counts().head(10).reset_index()
                top.columns = ["Value", "Count"]
                top["Percentage"] = (top["Count"] / len(df) * 100).round(2)
                st.dataframe(top, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("#### Least Frequent Values")
                bottom = col_data.value_counts().tail(10).reset_index()
                bottom.columns = ["Value", "Count"]
                bottom["Percentage"] = (bottom["Count"] / len(df) * 100).round(2)
                st.dataframe(bottom, use_container_width=True, hide_index=True)

    # ==================== VISUALIZE ====================
    with tabs[6]:
        perf.log("TAB: Visualize — start")
        _tab_t = time.perf_counter()
        st.markdown("### Visualizations")

        # ---- Data source selector ----
        viz_sources = ["parquet_data (original)"] + [
            f"{t} ({len(tdf):,} rows)" for t, tdf in st.session_state.temp_tables.items()
        ]
        viz_source_keys = ["parquet_data"] + list(st.session_state.temp_tables.keys())

        viz_sel = st.selectbox("Data Source", viz_sources, key="viz_source")
        viz_key = viz_source_keys[viz_sources.index(viz_sel)]

        if viz_key == "parquet_data":
            viz_df = df
        else:
            viz_df = st.session_state.temp_tables[viz_key]

        st.caption(f"Using **{viz_key}** — {len(viz_df):,} rows x {len(viz_df.columns)} cols")

        chart_type = st.selectbox("Chart Type", [
            "Histogram", "Bar Chart", "Scatter Plot", "Line Chart",
            "Box Plot", "Violin Plot", "Heatmap (Correlation)",
            "Pair Plot", "Pie Chart", "Treemap",
        ])

        numeric_cols = viz_df.select_dtypes("number").columns.tolist()
        all_cols = viz_df.columns.tolist()

        if chart_type == "Histogram":
            col = st.selectbox("Column", all_cols, key="hist_col")
            bins = st.slider("Bins", 10, 200, 50, key="hist_bins")
            color_col = st.selectbox("Color by", ["None"] + all_cols, key="hist_color")
            color = color_col if color_col != "None" else None
            fig = px.histogram(viz_df, x=col, nbins=bins, color=color, marginal="rug")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart":
            x_col = st.selectbox("X axis", all_cols, key="bar_x")
            agg = st.selectbox("Aggregation", ["count", "sum", "mean", "median"], key="bar_agg")
            y_col = st.selectbox("Y axis (for sum/mean/median)", ["None"] + numeric_cols, key="bar_y")
            if agg == "count":
                data = viz_df[x_col].value_counts().head(30).reset_index()
                data.columns = [x_col, "Count"]
                fig = px.bar(data, x=x_col, y="Count")
            elif y_col != "None":
                data = viz_df.groupby(x_col)[y_col].agg(agg).reset_index().head(30)
                fig = px.bar(data, x=x_col, y=y_col)
            else:
                st.warning("Select a Y axis column for this aggregation")
                fig = None
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X axis", numeric_cols, index=0, key="scat_x")
                y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scat_y")
                color_col = st.selectbox("Color by", ["None"] + all_cols, key="scat_color")
                size_col = st.selectbox("Size by", ["None"] + numeric_cols, key="scat_size")
                color = color_col if color_col != "None" else None
                size = size_col if size_col != "None" else None
                sample_n = min(5000, len(viz_df))
                fig = px.scatter(viz_df.sample(sample_n), x=x_col, y=y_col, color=color,
                                 size=size, opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for scatter plot.")

        elif chart_type == "Line Chart":
            x_col = st.selectbox("X axis", all_cols, key="line_x")
            y_cols = st.multiselect("Y axis", numeric_cols, default=numeric_cols[:1], key="line_y")
            if y_cols:
                fig = go.Figure()
                for y in y_cols:
                    fig.add_trace(go.Scatter(x=viz_df[x_col], y=viz_df[y], mode="lines", name=y))
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot":
            col = st.selectbox("Column", numeric_cols, key="box_col")
            group = st.selectbox("Group by", ["None"] + all_cols, key="box_group")
            color = group if group != "None" else None
            fig = px.box(viz_df, y=col, x=color, color=color)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Violin Plot":
            col = st.selectbox("Column", numeric_cols, key="vio_col")
            group = st.selectbox("Group by", ["None"] + all_cols, key="vio_group")
            color = group if group != "None" else None
            fig = px.violin(viz_df, y=col, x=color, color=color, box=True)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Heatmap (Correlation)":
            if len(numeric_cols) >= 2:
                method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_method")
                corr = viz_df[numeric_cols].corr(method=method)
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                aspect="auto", zmin=-1, zmax=1)
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns.")

        elif chart_type == "Pair Plot":
            if len(numeric_cols) >= 2:
                selected = st.multiselect("Columns (max 5)", numeric_cols,
                                          default=numeric_cols[:min(4, len(numeric_cols))], key="pair_cols")
                if selected and len(selected) <= 5:
                    sample_n = min(1000, len(viz_df))
                    fig = px.scatter_matrix(viz_df[selected].sample(sample_n), dimensions=selected, opacity=0.5)
                    fig.update_layout(height=800)
                    st.plotly_chart(fig, use_container_width=True)
                elif len(selected) > 5:
                    st.warning("Select at most 5 columns.")

        elif chart_type == "Pie Chart":
            col = st.selectbox("Column", all_cols, key="pie_col")
            top_n = st.slider("Top N categories", 3, 30, 10, key="pie_n")
            data = viz_df[col].value_counts().head(top_n)
            fig = px.pie(values=data.values, names=data.index.astype(str), hole=0.3)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Treemap":
            col = st.selectbox("Column", all_cols, key="tree_col")
            top_n = st.slider("Top N", 5, 50, 20, key="tree_n")
            data = viz_df[col].value_counts().head(top_n).reset_index()
            data.columns = ["Value", "Count"]
            fig = px.treemap(data, path=["Value"], values="Count")
            st.plotly_chart(fig, use_container_width=True)

    # ==================== ROW GROUPS ====================
    with tabs[7]:
        perf.log("TAB: Row Groups — start")
        _tab_t = time.perf_counter()
        st.markdown("### Row Group Details")

        for i, rg in enumerate(metadata["row_groups"]):
            with st.expander(f"Row Group {i} ({rg['num_rows']:,} rows, {rg['total_byte_size']:,} bytes)"):
                rg_data = []
                for col_info in rg["columns"]:
                    row = {
                        "Column": col_info["name"],
                        "Compression": col_info["compression"],
                        "Physical Type": col_info["physical_type"],
                        "Compressed Size": f"{col_info['total_compressed_size']:,}",
                        "Uncompressed Size": f"{col_info['total_uncompressed_size']:,}",
                        "Compression Ratio": f"{col_info['total_uncompressed_size'] / max(col_info['total_compressed_size'], 1):.2f}x",
                    }
                    if "min" in col_info:
                        row["Min"] = col_info["min"]
                        row["Max"] = col_info["max"]
                        row["Null Count"] = col_info["null_count"]
                    rg_data.append(row)
                st.dataframe(pd.DataFrame(rg_data), use_container_width=True, hide_index=True)

    # ==================== EXPORT ====================
    with tabs[8]:
        perf.log("TAB: Export — start")
        _tab_t = time.perf_counter()
        st.markdown("### Export Data")

        export_all_cols = st.checkbox("Export all columns", value=True, key="export_all_cols")
        if export_all_cols:
            export_cols = df.columns.tolist()
        else:
            export_cols = st.multiselect("Columns to export", df.columns.tolist(),
                                         default=df.columns[:5].tolist(), key="export_cols")
        export_rows = st.selectbox("Rows", ["All", "First N", "Last N", "Sample N"], key="export_rows")
        if export_rows != "All":
            n_export = st.number_input("N", min_value=1, max_value=len(df), value=min(1000, len(df)),
                                       key="export_n")
        else:
            n_export = len(df)

        export_df = df[export_cols]
        if export_rows == "First N":
            export_df = export_df.head(n_export)
        elif export_rows == "Last N":
            export_df = export_df.tail(n_export)
        elif export_rows == "Sample N":
            export_df = export_df.sample(min(n_export, len(export_df)))

        st.info(f"Will export {len(export_df):,} rows x {len(export_cols)} columns")

        EXCEL_MAX_ROWS = 1_048_576

        @st.cache_data
        def _to_csv(data):
            return data.to_csv(index=False)

        @st.cache_data
        def _to_json(data):
            return data.to_json(orient="records", date_format="iso")

        @st.cache_data
        def _to_excel(data):
            buf = io.BytesIO()
            data.head(EXCEL_MAX_ROWS).to_excel(buf, index=False, engine="openpyxl")
            return buf.getvalue(), len(data) > EXCEL_MAX_ROWS

        @st.cache_data
        def _to_parquet(data):
            buf = io.BytesIO()
            data.to_parquet(buf, index=False)
            return buf.getvalue()

        # Only prepare exports when user clicks the button
        if st.button("Prepare Export", type="primary", use_container_width=True):
            st.session_state.export_ready = True

        if st.session_state.get("export_ready"):
            with st.spinner("Preparing files..."):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.download_button("Download CSV", _to_csv(export_df), "export.csv", "text/csv",
                                       use_container_width=True)

                with col2:
                    st.download_button("Download JSON", _to_json(export_df), "export.json", "application/json",
                                       use_container_width=True)

                with col3:
                    excel_data, excel_truncated = _to_excel(export_df)
                    if excel_truncated:
                        st.caption(f"Excel limited to {EXCEL_MAX_ROWS:,} rows")
                    st.download_button("Download Excel", excel_data, "export.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       use_container_width=True)

                with col4:
                    st.download_button("Download Parquet", _to_parquet(export_df), "export.parquet",
                                       "application/octet-stream", use_container_width=True)

        st.markdown("#### Preview Export")
        st.dataframe(export_df.head(50), use_container_width=True, hide_index=True)

    # ==================== DEV LOGS ====================
    if st.session_state.dev_mode:
        with tabs[-1]:
            perf.log("TAB: Dev Logs — start")
            st.markdown("### Developer Performance Logs")

            total = perf.total_ms()

            # Summary metrics
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Total Rerun Time", f"{total:.0f} ms")
            dc2.metric("Log Entries", len(perf.entries))
            slow = [e for e in perf.entries if e["status"] == "slow"]
            dc3.metric("Slow Operations (>1s)", len(slow))

            if slow:
                st.error(f"**{len(slow)} slow operations detected!** These are taking >1 second each:")
                for s in slow:
                    st.markdown(f"- **{s['name']}** — {s['duration_ms']:.0f} ms (at {s['timestamp']})")

            st.markdown("---")

            # Full log table
            st.markdown("#### Execution Timeline")
            log_data = []
            for e in perf.entries:
                log_data.append({
                    "Offset": e["timestamp"],
                    "Section": e["name"],
                    "Duration (ms)": e["duration_ms"] if e["duration_ms"] > 0 else 0,
                    "Level": e["level"],
                    "Status": e["status"],
                })

            if log_data:
                log_df = pd.DataFrame(log_data)

                # Color-code status
                def _highlight_status(row):
                    if row["Status"] == "slow":
                        return ["background-color: rgba(248,113,113,0.2)"] * len(row)
                    elif row["Status"] == "error":
                        return ["background-color: rgba(248,113,113,0.4)"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    log_df.style.apply(_highlight_status, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )

            # Duration breakdown chart
            timed = [e for e in perf.entries if e["duration_ms"] > 0]
            if timed:
                st.markdown("#### Duration Breakdown")
                chart_df = pd.DataFrame([
                    {"Section": e["name"].strip(), "Duration (ms)": e["duration_ms"]}
                    for e in timed
                ]).sort_values("Duration (ms)", ascending=True)
                fig = px.bar(
                    chart_df, x="Duration (ms)", y="Section",
                    orientation="h", color="Duration (ms)",
                    color_continuous_scale=["#4ade80", "#fbbf24", "#f87171"],
                )
                fig.update_layout(height=max(300, len(timed) * 30), margin=dict(l=200))
                st.plotly_chart(fig, use_container_width=True)

            # Raw log (copyable)
            with st.expander("Raw Log (copyable)"):
                raw = "\n".join(
                    f"[{e['timestamp']}] [{e['status'].upper():7s}] {e['name']}"
                    + (f" ({e['duration_ms']}ms)" if e["duration_ms"] > 0 else "")
                    for e in perf.entries
                )
                st.code(raw, language="text")

            # Save to history
            st.session_state.perf_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "total_ms": total,
                "entries": len(perf.entries),
                "slow": len(slow),
            })

            # History chart
            if len(st.session_state.perf_history) > 1:
                st.markdown("#### Rerun History")
                hist_df = pd.DataFrame(list(st.session_state.perf_history))
                fig = px.line(hist_df, x="time", y="total_ms",
                              labels={"total_ms": "Total (ms)", "time": "Time"},
                              markers=True)
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
