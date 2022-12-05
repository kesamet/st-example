import numpy as np
import geopandas as gpd
import pandas as pd
import altair as alt
import pydeck as pdk
import streamlit as st
import h3
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import mapping, Polygon

from src.data import s2
from src.constants import CITIES

st.set_page_config(layout="wide")


@st.experimental_memo
def load_csv(filename, **kwargs):
    try:
        return pd.read_csv(filename, **kwargs)
    except FileNotFoundError:
        pass


@st.experimental_memo
def load_embeddings(filename):
    try:
        return np.load(filename)
    except FileNotFoundError:
        pass


@st.experimental_memo
def pca_scale(embeddings):
    pca_embeddings = PCA(n_components=3).fit_transform(embeddings)
    rgb = MinMaxScaler().fit_transform(pca_embeddings)
    rgb = np.clip(rgb, 0, 1)
    return rgb


def _h3_to_polygon(hex_id: str) -> Polygon:
    boundary = h3.h3_to_geo_boundary(hex_id)
    boundary = [[y, x] for [x, y] in boundary]
    h3_polygon = Polygon(boundary)
    return h3_polygon


def _s2_to_polygon(cell) -> Polygon:
    if isinstance(cell, dict):
        boundary = cell["geometry"]
    elif isinstance(cell, str):
        boundary = s2.s2_to_geo_boundary(cell)
    else:
        boundary = cell
    boundary = [[y, x] for [x, y] in boundary]
    s2_polygon = Polygon(boundary)
    return s2_polygon


@st.experimental_memo
def _cast_gdf(cell_type, df, fill_color):
    _df = df.copy()

    if cell_type == "h3":
        _df["geometry"] = _df["h3"].apply(_h3_to_polygon)
    else:
        _df["geometry"] = _df["s2"].apply(_s2_to_polygon)

    _df = gpd.GeoDataFrame(_df, crs="EPSG:4326")
    lng = _df["geometry"].centroid.x.mean().item()
    lat = _df["geometry"].centroid.y.mean().item()

    _df["fill_color"] = [(c * 255).astype(int) for c in fill_color]
    # df0 = pd.read_json(_df[["h3", "fill_color"]].to_json())  # HACK
    _df["coordinates"] = _df["geometry"].apply(lambda x: mapping(x)["coordinates"][0])
    df0 = pd.read_json(_df[["coordinates", "fill_color"]].to_json())  # HACK
    return (lat, lng), df0


def add_map(centre, df0):
    initial_view_state = pdk.ViewState(
        latitude=centre[0],
        longitude=centre[1],
        zoom=10,
        pitch=0,
    )
    # polygon_layer = pdk.Layer(
    #     "H3HexagonLayer",
    #     df0,
    #     pickable=True,
    #     stroked=True,
    #     filled=True,
    #     extruded=False,
    #     opacity=0.2,
    #     get_hexagon="h3",
    #     get_fill_color="fill_color",
    # )
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        df0,
        stroked=False,
        opacity=0.4,
        get_polygon="coordinates",
        get_fill_color="fill_color",
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=initial_view_state,
            layers=[polygon_layer],
        )
    )


def page_analysis():
    models = {
        "Autoencoder - H3": {
            "cell_type": "h3",
            "embeddings": "data/embeddings_h3_9_v13.npy",
        },
        "Binary - H3": {
            "cell_type": "h3",
            "embeddings": "data/embeddings_h3_9_v10.npy",
        },
        "Autoencoder - S2": {
            "cell_type": "s2",
            "embeddings": "data/embeddings_s2_14_v15.npy",
        },
    }
    v = st.sidebar.selectbox("Version", list(models.keys()))
    embeddings = load_embeddings(models[v]["embeddings"])
    h3_df = load_csv("data/h3_9_city.csv")
    s2_df = load_csv("data/s2_14_city.csv")

    select_city = st.sidebar.selectbox("Select city", list(CITIES.keys()))
    if models[v]["cell_type"] == "h3":
        select_regions = h3_df["city"] == select_city
        _df = h3_df[select_regions]
    else:
        select_regions = s2_df["city"] == select_city
        _df = s2_df[select_regions]

    fill_color = pca_scale(embeddings[select_regions])

    centre, df0 = _cast_gdf(models[v]["cell_type"], _df, fill_color)
    add_map(centre, df0)


def main():
    st.sidebar.title("Spatial embeddings")
    with st.sidebar.expander("Notes"):
        st.info(
            "Details found [here](https://go-jek.atlassian.net/wiki/spaces/~6201d607a2940200687856c0/pages/2571933488/Location+embeddings)"
        )

    # pages = {
    #     "Analysis": page_analysis,
    # }
    # select_page = st.sidebar.radio("pages", list(pages.keys()), label_visibility="hidden")
    # st.header(select_page)
    # pages[select_page]()
    page_analysis()


if __name__ == "__main__":
    main()
