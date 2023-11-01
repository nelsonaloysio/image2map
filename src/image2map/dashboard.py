from io import BytesIO
from time import time

import numpy as np
import streamlit as st
from PIL import Image

from .app import BaseApp
from .som import SOM
from .utils import minmax_scale, split_ts

PAGE_CONFIG = {
    "layout": "wide",
    "page_title": "Kohonen Self-Organizing Map",
    "page_icon": "🤖",
    "menu_items": {},
}

TITLE = """
<h1 align="center">✨ Kohonen Self-Organizing Map ✨</h1>
"""


class Dashboard(BaseApp):

    def __init__(self) -> None:
        st.set_page_config(**PAGE_CONFIG)

    def __call__(self) -> None:
        st.write(TITLE, unsafe_allow_html=True)
        st.write("___")

        content = st.container()

        with st.sidebar:
            self.title("Image2Map", h=3)
            st.write("This application allows you to train a Kohonen Self-Organizing Map (SOM) on a set of images.")
            st.write("Upload files and configure the SOM hyperparameters to train and visualize the results.")
            st.write("Source code on [GitHub](https://github.com/nelsonaloysio/image2map).")

        with content:
            if st.session_state.get("som", None) is not None:
                self.display_som()

            else:
                self.display_uploader()

                if "data" in st.session_state:
                    self.title("Shape configuration", h=5)
                    st.write("Select the parameters for the SOM training below:")
                    self.display_configure()

    def display_som(self) -> None:
        st.title("Results")

        st.write(f"Finished training SOM in `{st.session_state.time_elapsed:.3f}` seconds "
                 f"for `{st.session_state.config['epochs']}` epochs with the parameters:")

        st.write(st.session_state.som)

        self.title("Output map", h=3)
        st.write("The output map shows the weights of the neurons in the SOM after training. "
                 "Scroll down to find the best matching unit for a new input and visualize the results.")

        if st.session_state.get("data_test") is not None:
            st.info(f"Best matching unit (BMU) for the uploaded image is: `{st.session_state.j0}`.")
            image = st.session_state.X_test

            if st.session_state.config["topology"] in ("GRID", "MESH"):
                image = image.reshape(16, 16)

            st.write("Currently uploaded image (maximum of 100 pixels of width):")
            st.image(image, width=100)

        c1, c2 = st.columns(2)

        self.title("Find best matching unit", h=5)
        st.write("Upload a new image to test the trained SOM.")
        self.uploader("data_test", accept_multiple_files=False)

        if st.session_state.get("data_test") is not None:
            X_test = np.array([
                Image.open(BytesIO(x)).convert("L").getdata()
                for x in st.session_state.data_test
            ])
            st.session_state.j0 = st.session_state.som.predict(X_test)[0]
            st.session_state.X_test = X_test

        with c1:
            self.title("Unit topology", h=5)
            st.write(st.session_state.som.plot_neurons(j0=st.session_state.get("j0", None)))
        with c2:
            self.title("Unit weights", h=5)
            st.write(st.session_state.som.plot_weights(j0=st.session_state.get("j0", None)))

        self.title("Reset neural network", h=5)
        st.write("Restart training from scratch.")
        if st.button("Reset"):
            st.session_state.som = None
            st.rerun()

    def display_uploader(self) -> None:
        self.title("File uploader", h=5)
        self.uploader()

    def display_configure(self) -> None:
        c1, c2 = st.columns(2)
        disabled = False

        with c1:
            # krow
            st.slider(
                "Number of rows in the output grid",
                min_value=1,
                max_value=100,
                value=10,
                key="krow",
                help="Number of rows in the output grid (default: 10)."
            )
            # topology
            st.selectbox(
                "Topology",
                options=["GRID", "MESH", "LINE", "RING"],
                index=0,
                key="topology",
                help="Topology of the SOM (default: GRID)."
            )
        with c2:
            st.slider(
                "Number of columns in the output grid",
                min_value=1,
                max_value=100,
                value=st.session_state.krow if st.session_state.get("topology") in ("GRID", "MESH") else 1,
                key="kcol",
                disabled=True if st.session_state.get("topology") in ("LINE", "RING") else False,
                help="Number of columns in the output grid (default: 10)."
            )
            # topology
            st.selectbox(
                "Unit (neuron) topology",
                options=["SQUARE", "HEXAGONAL"],
                index=0,
                key="unit_topology",
                disabled=True,
                help="Topology for neighborhood function (default: SQUARE)."
            )

        if st.session_state.get("topology") in ("RING", "MESH"):
            st.warning("Topology not implemented, expects ('GRID', 'LINE').")
            disabled = True

        if st.session_state.get("topology") in ("RING", "MESH")\
        and st.session_state.krow != st.session_state.kcol:
            st.warning("Incompatible grid shape for selected topology, expects (k, k) shape.")
            disabled = True

        self.title("Training parameters", h=5)
        st.write("Configure the training parameters for the SOM below:")
        c1, c2 = st.columns(2)

        with c1:
            # epochs
            st.slider(
                "Number of training epochs",
                min_value=1,
                max_value=1000,
                value=100,
                key="epochs",
                help="Number of training epochs (default: 100).",
                disabled=disabled
            )
            # alpha_min
            st.slider(
                "Minimum learning rate",
                min_value=0.01,
                max_value=1.0,
                value=0.01,
                key="alpha_min",
                help="Minimum learning rate (default: 0.01).",
                disabled=disabled
            )
            # radius_min
            st.slider(
                "Minimum radius for neighborhood function",
                min_value=0,
                max_value=100,
                value=0,
                key="radius_min",
                help="Minimum radius for neighborhood function (default: 0).",
                disabled=disabled
            )
        with c2:
            # epochs
            st.slider(
                "Training set size",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                key="ts_size",
                help="Size of the training set as a fraction of the total dataset (default: 1.0).",
                disabled=disabled
            )
            # alpha_max
            st.slider(
                "Maximum learning rate",
                min_value=0.01,
                max_value=1.0,
                value=1.0,
                key="alpha_max",
                help="Maximum learning rate (default: 1.0).",
                disabled=disabled
            )
            # radius_max
            st.slider(
                "Maximum radius for neighborhood function (optional, 0 to auto-select)",
                min_value=0,
                max_value=20,
                value=0,
                key="radius_max",
                help="Maximum radius for neighborhood function (optional, 0 to auto-select).",
                disabled=disabled
            )

        self.title("Train model", h=5)
        st.write("Click the button below to train the SOM with the selected parameters.")
        if st.button("Train", disabled=disabled):
            with st.spinner("Training neural network..."):
                self.train()

        # self.title("Visualize map", h=5)
        # st.write("Click the button below to visualize the SOM.")
        # if st.button("Visualize", disabled=disabled):
        #     self.visualize()
        #     st.write("Success!")

        # self.title("Predict output", h=5)
        # st.write("Upload a new file and obtain best matching unit (BMU).")
        # self.uploader("predict_form", "predict")

        # if st.button("Visualize", disabled=disabled):
        #     self.visualize()
        #     st.write("Success!")

    def train(self) -> None:
        k_units = st.session_state.krow
        if st.session_state.topology in ("GRID", "MESH"):
            k_units *= st.session_state.kcol

        k_shape = (st.session_state.krow, st.session_state.kcol)
        if st.session_state.topology in ("LINE", "RING"):
            k_shape = k_units

        som = SOM(
            k_units=k_units,
            k_shape=k_shape,
            k_dist=st.session_state.get("k_dist", "l2"),
            n_inputs=st.session_state.get("n_inputs",  None),
            n_shape=st.session_state.get("n_shape",  None),
            topology=st.session_state.get("topology", "grid",),
            unit_topology=st.session_state.get("unit_topology", "square",),
            radius_max=st.session_state.get("radius_max",  None) or None,
            radius_min=st.session_state.get("radius_min",  0),
            radius_rate=st.session_state.get("radius_rate",  None),
            radius_decay=st.session_state.get("radius_decay", "exp",),
            alpha_max=st.session_state.get("alpha_max", .1),
            alpha_min=st.session_state.get("alpha_min", .01),
            alpha_rate=st.session_state.get("alpha_rate",  None),
            alpha_decay=st.session_state.get("alpha_decay", "exp",),
            phi=st.session_state.get("phi", "lap",),
            k=st.session_state.get("k", 1.0),
            sigma=st.session_state.get("sigma", 1.0),
            seed=st.session_state.get("seed",  None),
        )

        try:
            X = np.array([
                Image.open(BytesIO(x)).convert("L").getdata()
                for x in st.session_state.data
            ])
        except:
            st.error("Error processing uploaded files, please try again.")
            return

        X = minmax_scale(X, (-1, 1))
        X_train, _ = split_ts(X, st.session_state.ts_size)

        som.fit(X_train)
        som.init_neurons()
        som.init_weights()

        t0 = time()
        som.train(X_train, epochs=st.session_state.epochs)
        st.session_state.time_elapsed = time() - t0

        st.session_state.config = {
            "krow": st.session_state.krow,
            "topology": st.session_state.topology,
            "kcol": st.session_state.kcol,
            "unit_topology": st.session_state.unit_topology,
            "epochs": st.session_state.epochs,
            "alpha_min": st.session_state.alpha_min,
            "radius_min": st.session_state.radius_min,
            "ts_size": st.session_state.ts_size,
            "alpha_max": st.session_state.alpha_max,
            "radius_max": st.session_state.radius_max,
        }

        st.session_state.som = som
        st.rerun()