"""Read-oriented Streamlit client for the authenticated native API."""

import json
import os
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Intelligence", layout="wide")
st.title("Football Data Analytics Software")
st.caption("Senior men's A-internationals · Regulation-time forecasts · Auditable research")
base = os.getenv("FOOTBALL_API_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
key = st.sidebar.text_input(
    "API access key", value=os.getenv("FOOTBALL_API_KEY", ""), type="password"
)
page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Ratings",
        "Match forecasts",
        "Match history",
        "Forecast batches",
        "Team intelligence",
        "Model diagnostics",
        "Post-match review",
    ],
)


def fetch(path: str):
    headers = {"Authorization": "Bearer " + key} if key else {}
    with urlopen(Request(base + path, headers=headers), timeout=10) as response:
        return json.load(response)


def paged(
    path: str, label: str, *, initial_as_of: str | None = None,
    cursor_param: str = "after", limit: int | None = None,
):
    # Timestamp and committed sequence keep Previous/Next on the same snapshot.
    state_key = "collection:" + base + ":" + path
    if initial_as_of is not None:
        state_key += ":" + initial_as_of
    if state_key not in st.session_state:
        st.session_state[state_key] = {"cursors": [None], "as_of": initial_as_of}
    state = st.session_state[state_key]
    params = {}
    if limit is not None:
        params["limit"] = limit
    if state["cursors"][-1] is not None:
        params[cursor_param] = state["cursors"][-1]
    if state["as_of"] is not None:
        params["as_of"] = state["as_of"]
    if state.get("through_sequence") is not None:
        params["through_sequence"] = state["through_sequence"]
    response = fetch(path + ("?" + urlencode(params) if params else ""))
    state["as_of"] = response.get("as_of")
    state["through_sequence"] = response.get("through_sequence")
    previous, following, refresh = st.columns(3)
    if previous.button("Previous " + label, disabled=len(state["cursors"]) == 1):
        state["cursors"].pop()
        st.rerun()
    cursor = response.get("next_cursor")
    if following.button("Next " + label, disabled=not cursor):
        if cursor in state["cursors"]:
            raise ValueError("API returned a repeated cursor")
        state["cursors"].append(cursor)
        st.rerun()
    if refresh.button("Refresh " + label):
        del st.session_state[state_key]
        st.rerun()
    st.caption(f"{label.capitalize()} · page {len(state['cursors'])} · "
               f"{len(response['items'])} returned")
    return response


try:
    if page == "Overview":
        st.json(fetch("/health"))
        teams, models = paged("/teams", "teams"), paged("/models", "models")
        a, b = st.columns(2)
        a.metric("Teams returned", len(teams["items"]))
        b.metric("Registered models returned", len(models["items"]))
        st.info(
            "Model approval and source licensing are separate from predictive quality. "
            "Probability intervals remain unvalidated; synthetic demos do not establish skill."
        )
        st.dataframe(pd.DataFrame(models["items"]), hide_index=True)
    elif page == "Ratings":
        rows = fetch("/ratings")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(
            "Glicko deviation rises with inactivity. "
            "Intervals describe ratings, not match probabilities."
        )
        if rows:
            st.bar_chart(pd.DataFrame(rows).set_index("team_id")[["rating"]])
    elif page == "Match forecasts":
        forecasts = paged("/predictions", "forecasts")
        choices = {f["forecast_id"]: f for f in forecasts["items"]}
        if choices:
            chosen = st.selectbox("Forecast", list(choices))
            forecast = choices[chosen]
            st.subheader(
                f"{forecast['match']['home_team_id']} vs {forecast['match']['away_team_id']}"
            )
            st.bar_chart(pd.Series(forecast["probabilities"], name="Probability"))
            st.write("Model", forecast["bundle_id"], "Method", forecast["method"])
            st.write("Training snapshot", forecast["data_timestamp"])
            st.caption(forecast["explanation_note"])
            st.dataframe(pd.DataFrame(forecast["explanatory_drivers"]), hide_index=True)
            with st.expander("Complete forecast and lineage"):
                st.json(forecast)
            if st.button("Inspect forecast inputs", key="inspect-lineage:" + chosen):
                lineage = fetch("/predictions/" + quote(chosen, safe="") + "/lineage")
                if lineage["forecast_id"] != chosen:
                    raise ValueError("Lineage response belongs to a different forecast")
                if lineage["status"] == "resolved":
                    st.subheader("Forecast input revisions")
                    st.caption(
                        "These are the exact historical inputs. Resolution does not establish "
                        "current model approval or validated predictive performance."
                    )
                    st.dataframe(pd.DataFrame(lineage["records"]), hide_index=True)
                    st.download_button(
                        "Download lineage report", json.dumps(lineage, indent=2),
                        file_name=chosen + "-lineage.json", mime="application/json",
                    )
                elif lineage["status"] == "unavailable":
                    st.info("Exact input revisions were not recorded for this forecast.")
                else:
                    raise ValueError("Unknown lineage resolution status")
        else:
            st.info("No forecasts have been stored.")
    elif page == "Match history":
        matches = paged("/matches", "matches")["items"]
        choices = {row["match"]["match_id"]: row for row in matches}
        if choices:
            selected = st.selectbox("Match", list(choices))
            cutoff_text = st.text_input(
                "Historical cutoff (UTC)",
                help="Optional ISO timestamp with timezone, for example 2020-06-01T12:00:00+00:00",
            )
            cutoff = None
            if cutoff_text.strip():
                try:
                    parsed = datetime.fromisoformat(cutoff_text.strip().replace("Z", "+00:00"))
                    if parsed.tzinfo is None or parsed.utcoffset() is None:
                        raise ValueError("Historical cutoff requires a timezone")
                    cutoff = parsed.astimezone(UTC).isoformat()
                except ValueError:
                    st.error("Enter an ISO timestamp with a timezone, such as "
                             "2020-06-01T12:00:00+00:00.")
                    st.stop()
            history = paged(
                "/matches/" + quote(selected, safe="") + "/history", "revisions",
                initial_as_of=cutoff, cursor_param="after_sequence", limit=25,
            )
            st.caption("Revisions are ordered by commit sequence. Later imports may be backdated.")
            st.write("Snapshot", history["as_of"])
            revisions = history["items"]
            if revisions:
                st.dataframe(pd.DataFrame([
                    {"sequence": row["sequence"], "record_id": row["record_id"],
                     "available_at": row["available_at"], "recorded_at": row["recorded_at"],
                     "status": row["payload"]["match"].get("status"),
                     "home_score": row["payload"].get("home_score"),
                     "away_score": row["payload"].get("away_score")}
                    for row in revisions
                ]), hide_index=True)
                by_id = {row["record_id"]: row for row in revisions}
                revision = st.selectbox("Revision", list(by_id))
                with st.expander("Stored revision and source provenance", expanded=True):
                    st.json(by_id[revision])
            else:
                st.info("No revisions are visible at this historical cutoff.")
        else:
            st.info("No matches have been imported.")
    elif page == "Forecast batches":
        batches = paged("/forecast-batches", "batches")["items"]
        if batches:
            choices = {r["run_id"]: r for r in batches}
            selected = st.selectbox("Batch run", list(choices))
            report = choices[selected]
            st.write("Status", report["status"], "Snapshot", report["snapshot_at"])
            st.json(report["counts"])
            st.dataframe(pd.DataFrame(report["fixtures"]), hide_index=True)
            with st.expander("Request, review lineage and errors"):
                st.json(report)
        else:
            st.info("No forecast batches have been stored.")
    elif page == "Team intelligence":
        teams = paged("/teams", "teams")["items"]
        if teams:
            team = st.selectbox("Team", [row["team_id"] for row in teams])
            report = fetch("/reports/team/" + quote(team, safe=""))
            st.json(report["rating"])
            st.dataframe(pd.DataFrame(report["recent_form"]), hide_index=True)
            st.caption(report["watch_note"])
            if report["rating_history"]:
                st.line_chart(
                    pd.DataFrame(report["rating_history"]).set_index("effective_date")[["rating"]]
                )
    elif page == "Model diagnostics":
        models = paged("/models", "models")["items"]
        if models:
            model = st.selectbox("Model", [row["model_id"] for row in models])
            report = fetch("/models/" + quote(model, safe="") + "/metrics")
            rows = report.get("comparison", {}).get("rows", []) if report.get("comparison") else []
            st.dataframe(pd.json_normalize(rows), hide_index=True)
            for run in report.get("runs", []):
                with st.expander(run["backtest"]["model_spec_id"]):
                    st.json(run["calibration"])
            st.caption("Metrics describe linked chronological research, not serving performance.")
    else:
        st.dataframe(pd.DataFrame(fetch("/diagnostics/upsets")), hide_index=True)
        st.caption(
            "Surprisal ranks realized outcomes by forecast probability, without causal attribution."
        )
except (HTTPError, URLError, TimeoutError, ValueError) as error:
    st.error(
        f"API unavailable or request rejected ({type(error).__name__}). "
        "Check service status, access key and whether the requested data has been imported."
    )
