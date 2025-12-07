import os
import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================

DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "monthly_budget.db")


# =========================
# DB HELPERS
# =========================

def get_conn():
    """Ensure data folder exists and return a SQLite connection."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    """Create or upgrade the monthly_budget table."""
    conn = get_conn()

    # Base table (original columns)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS monthly_budget (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month_index INTEGER NOT NULL,
            month_name TEXT NOT NULL,
            year INTEGER NOT NULL,
            gross_salary REAL,
            net_salary REAL,
            taxes REAL,
            deductions REAL,
            recurrent_bills REAL
        );
        """
    )

    # Check existing columns
    cur = conn.execute("PRAGMA table_info(monthly_budget);")
    existing_cols = [row[1] for row in cur.fetchall()]

    # Add other_spend column if missing
    if "other_spend" not in existing_cols:
        conn.execute("ALTER TABLE monthly_budget ADD COLUMN other_spend REAL;")

    conn.commit()
    conn.close()


def upsert_month_row(
    month_index: int,
    month_name: str,
    year: int,
    gross_salary: float,
    net_salary: float,
    taxes: float,
    deductions: float,
    recurrent_bills: float,
    other_spend: float,
):
    """
    Insert a new row for (month_index, year) or update it if it already exists.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Does this month/year already exist?
    cur.execute(
        """
        SELECT id
        FROM monthly_budget
        WHERE month_index = ? AND year = ?
        """,
        (month_index, year),
    )
    row = cur.fetchone()

    if row:
        # Update existing row
        cur.execute(
            """
            UPDATE monthly_budget
            SET month_name = ?,
                gross_salary = ?,
                net_salary = ?,
                taxes = ?,
                deductions = ?,
                recurrent_bills = ?,
                other_spend = ?
            WHERE id = ?
            """,
            (
                month_name,
                gross_salary,
                net_salary,
                taxes,
                deductions,
                recurrent_bills,
                other_spend,
                row[0],
            ),
        )
    else:
        # Insert new row (9 columns -> 9 placeholders)
        cur.execute(
            """
            INSERT INTO monthly_budget (
                month_index, month_name, year,
                gross_salary, net_salary, taxes,
                deductions, recurrent_bills, other_spend
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                month_index,
                month_name,
                year,
                gross_salary,
                net_salary,
                taxes,
                deductions,
                recurrent_bills,
                other_spend,
            ),
        )

    conn.commit()
    conn.close()


@st.cache_data
def load_monthly_budget() -> pd.DataFrame:
    """Pretty dataframe for display."""
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            month_index AS "Month Index",
            month_name  AS "Month",
            year        AS "Year",
            gross_salary AS "Gross Salary",
            taxes        AS "Taxes",
            deductions   AS "Deductions",
            recurrent_bills AS "Recurrent Bills",
            other_spend AS "Other Spend"
        FROM monthly_budget
        ORDER BY Year DESC, "Month Index" DESC
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return df

    df["Net Salary (after bills)"] = (
        df["Gross Salary"]
        - df["Taxes"]
        - df["Deductions"]
        - df["Recurrent Bills"]
        - df["Other Spend"]
    )

    return df


@st.cache_data
def load_monthly_budget_raw() -> pd.DataFrame:
    """Raw table (DB column names) for manual editing."""
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM monthly_budget", conn)
    conn.close()
    return df


def update_single_field(row_id: int, column: str, new_value):
    """
    Update a single column for a given row.
    If the column affects net_salary, recompute net_salary too.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Fetch existing row to recompute net if needed
    cur.execute("SELECT * FROM monthly_budget WHERE id = ?", (row_id,))
    row = cur.fetchone()
    col_names = [d[0] for d in cur.description]
    row_dict = dict(zip(col_names, row))

    numeric_cols = {
        "gross_salary",
        "taxes",
        "deductions",
        "recurrent_bills",
        "other_spend",
        "net_salary",
    }

    if column in numeric_cols:
        new_value_casted = float(new_value)
    elif column in ["month_index", "year"]:
        new_value_casted = int(new_value)
    else:
        new_value_casted = str(new_value)

    # Recompute net_salary if needed
    if column in ["gross_salary", "taxes", "deductions", "recurrent_bills", "other_spend"]:
        gross = new_value_casted if column == "gross_salary" else (row_dict.get("gross_salary") or 0.0)
        taxes = new_value_casted if column == "taxes" else (row_dict.get("taxes") or 0.0)
        deductions = new_value_casted if column == "deductions" else (row_dict.get("deductions") or 0.0)
        bills = new_value_casted if column == "recurrent_bills" else (row_dict.get("recurrent_bills") or 0.0)
        other = new_value_casted if column == "other_spend" else (row_dict.get("other_spend") or 0.0)

        new_net = gross - taxes - deductions - bills - other

        cur.execute(
            f"UPDATE monthly_budget SET {column} = ?, net_salary = ? WHERE id = ?",
            (new_value_casted, new_net, row_id),
        )
    else:
        cur.execute(
            f"UPDATE monthly_budget SET {column} = ? WHERE id = ?",
            (new_value_casted, row_id),
        )

    conn.commit()
    conn.close()


# =========================
# STREAMLIT PAGE
# =========================

init_db()

st.title("💰 Monthly Budget Tracker")

st.write(
    "Track each month's salary, deductions, fixed bills, and estimated spending. "
    "Use the tabs below to view, add/update, or manually edit records."
)

today = date.today()
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ---------- TABS ----------
tab_overview, tab_add, tab_edit = st.tabs(
    ["Overview", "Add / Update Month", "Manual Edit"]
)

# ---------- OVERVIEW TAB ----------
with tab_overview:
    st.subheader("Saved Monthly Budgets")

    df = load_monthly_budget()
    if df.empty:
        st.info("No months saved yet. Add your first month in the next tab.")
    else:
        st.dataframe(
            df.style.format(
                {
                    "Gross Salary": "${:,.0f}",
                    "Taxes": "${:,.0f}",
                    "Deductions": "${:,.0f}",
                    "Recurrent Bills": "${:,.0f}",
                    "Other Spend": "${:,.0f}",
                    "Net Salary (after bills)": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )

# ---------- ADD / UPDATE TAB ----------
with tab_add:
    st.subheader("Add or Update a Monthly Record")

    # --- Month & Year selection ---
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        year = st.number_input(
            "Year",
            min_value=2020,
            max_value=2100,
            value=today.year,
            step=1,
        )
    with col_date2:
        month_name = st.selectbox(
            "Month",
            options=month_names,
            index=today.month - 1,
        )
    month_index = month_names.index(month_name) + 1

    # --- Income & deductions ---
    st.markdown("### Income & Deductions")

    c1, c2 = st.columns(2)
    with c1:
        gross_salary = st.number_input(
            "Gross Monthly Salary",
            min_value=0.0,
            step=100.0,
        )
        taxes = st.number_input(
            "Taxes (total for the month)",
            min_value=0.0,
            step=50.0,
        )
    with c2:
        deductions = st.number_input(
            "Other Deductions (retirement, medical, etc.)",
            min_value=0.0,
            step=50.0,
        )

    # --- Fixed bills ---
    st.markdown("### Fixed Monthly Bills")

    recurrent_bills = st.number_input(
        "Recurrent Bills (rent, utilities, subscriptions, insurance, etc.)",
        min_value=0.0,
        step=25.0,
    )

    # --- Estimated Spend ---
    st.markdown("### Estimated Monthly Spend")

    other_spend = st.number_input(
        "Food, Tech, Hobbies, Nala",
        min_value=0.0,
        step=25.00,
    )

    # --- Calculate net after everything ---
    net_after_bills = (
        float(gross_salary)
        - float(taxes)
        - float(deductions)
        - float(recurrent_bills)
        - float(other_spend)
    )

    st.markdown("---")
    st.subheader("Net Salary After Taxes, Deductions, Bills & Estimated Spend")
    st.metric(
        "Net Salary (after everything)",
        f"${net_after_bills:,.2f}",
    )
    st.markdown("---")

    # --- Save button ---
    if st.button("💾 Save / Update This Month"):
        upsert_month_row(
            month_index,
            month_name,
            int(year),
            float(gross_salary),
            float(net_after_bills),  # store calculated net
            float(taxes),
            float(deductions),
            float(recurrent_bills),
            float(other_spend),
        )
        load_monthly_budget.clear()
        load_monthly_budget_raw.clear()
        st.success(f"Budget saved for {month_name} {year}.")

# ---------- MANUAL EDIT TAB ----------
with tab_edit:
    st.subheader("Manual Edit of Existing Rows")

    raw_df = load_monthly_budget_raw()
    if raw_df.empty:
        st.info("No rows to edit yet. Save a month in the 'Add / Update Month' tab first.")
    else:
        # Build a label like "2025-03 (March)"
        raw_df["label"] = raw_df.apply(
            lambda r: f"{int(r['year'])}-{int(r['month_index']):02d} ({r['month_name']})",
            axis=1,
        )

        selected_label = st.selectbox(
            "Select month/year to edit",
            options=raw_df["label"],
        )

        row = raw_df.loc[raw_df["label"] == selected_label].iloc[0]
        row_id = int(row["id"])

        st.write("**Current row values:**")
        st.dataframe(
            {
                "month_index": int(row["month_index"]),
                "month_name": row["month_name"],
                "year": int(row["year"]),
                "gross_salary": row.get("gross_salary"),
                "taxes": row.get("taxes"),
                "deductions": row.get("deductions"),
                "recurrent_bills": row.get("recurrent_bills"),
                "other_spend": row.get("other_spend"),
                "net_salary": row.get("net_salary"),
            }
        )

        editable_cols = [
            "month_index",
            "month_name",
            "year",
            "gross_salary",
            "taxes",
            "deductions",
            "recurrent_bills",
            "other_spend",
            "net_salary",
        ]

        col_to_edit = st.selectbox(
            "Which column do you want to edit?",
            options=editable_cols,
        )

        current_value = row[col_to_edit]
        st.write(f"Current value: `{current_value}`")

        numeric_cols = {
            "gross_salary",
            "taxes",
            "deductions",
            "recurrent_bills",
            "other_spend",
            "net_salary",
        }

        if col_to_edit in numeric_cols:
            new_value = st.number_input(
                "New value",
                value=float(current_value) if current_value is not None else 0.0,
            )
        elif col_to_edit in ["month_index", "year"]:
            new_value = st.number_input(
                "New value",
                value=int(current_value) if current_value is not None else 1,
                step=1,
            )
        else:
            new_value = st.text_input(
                "New value",
                value=str(current_value) if current_value is not None else "",
            )

        if st.button("Update selected field"):
            update_single_field(row_id, col_to_edit, new_value)
            load_monthly_budget.clear()
            load_monthly_budget_raw.clear()
            st.success(
                f"Updated {col_to_edit} for {selected_label} to {new_value} "
                "(net_salary recomputed if applicable)."
            )
