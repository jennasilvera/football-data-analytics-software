"""Read-oriented Streamlit client for the authenticated native API."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Intelligence", layout="wide")
st.title("Football Data Analytics Software")
st.caption("Senior men's A-internationals · Regulation-time forecasts · Auditable research")
base = os.getenv("FOOTBALL_API_URL", "http://127.0.0.1:8000").rstrip("/")
key = st.sidebar.text_input(
    "API access key", value=os.getenv("FOOTBALL_API_KEY", ""), type="password"
)
page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Ratings",
        "Match forecasts",
        "Team intelligence",
        "Model diagnostics",
        "Post-match review",
    ],
)


def fetch(path: str):
    headers = {"Authorization": "Bearer " + key} if key else {}
    with urlopen(Request(base + path, headers=headers), timeout=10) as response:
        return json.load(response)


try:
    if page == "Overview":
        st.json(fetch("/health"))
        teams, models = fetch("/teams"), fetch("/models")
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
        forecasts = fetch("/predictions")
        if forecasts["truncated"]:
            st.warning("The API returned a bounded page. More forecasts exist.")
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
        else:
            st.info("No forecasts have been stored.")
    elif page == "Team intelligence":
        teams = fetch("/teams")["items"]
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
        models = fetch("/models")["items"]
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
