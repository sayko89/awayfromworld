import importlib
import streamlit as st

st.set_page_config(page_title="Exoplanet ML Pipelines", layout="wide")

st.title("🔭 Exoplanet ML Pipelines")
st.write("Select a pipeline below to run its step-by-step placeholder. "
         "You can plug in real ML code inside each module's `main()` later.")

# Safe-caller for modules' main()
def run_module(module_name: str, pretty_name: str):
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "main") and callable(module.main):
            st.info(f"Launching **{pretty_name}** pipeline...", icon="ℹ️")
            module.main()
            st.success(f"{pretty_name} completed (placeholder).", icon="✅")
        else:
            st.warning(f"`main()` not found in `{module_name}`. "
                       f"Add a `def main(): ...` function to enable this section.", icon="⚠️")
    except Exception as e:
        st.exception(e)

with st.container():
    st.header("K2 Analysis")
    if st.button("▶️ Run K2 Analysis", key="btn_k2"):
        run_module("ml_models.k2", "K2 Analysis")

st.divider()

with st.container():
    st.header("NASA TOI Analysis")
    if st.button("▶️ Run NASA TOI Analysis", key="btn_toi"):
        run_module("ml_models.nasa_toi", "NASA TOI Analysis")

st.divider()

with st.container():
    st.header("NASA Kepler Analysis")
    if st.button("▶️ Run NASA Kepler Analysis", key="btn_kepler"):
        run_module("ml_models.nasakepler", "NASA Kepler Analysis")