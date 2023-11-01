from abc import ABCMeta, abstractmethod

import streamlit as st

CACHE_DATA = {
    "ttl": 1800,
}

CACHE_RESOURCE = {
    "ttl": 1800,
}

CACHE_TO_DISK = {
    "max_entries": 100,
    "persist": "disk",
}


class BaseApp(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        """ Initializes class. """

    def uploader(self, key: str = "data", accept_multiple_files: bool = True) -> None:
        with st.form(f"uploader_{key}_form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Please select one or more images to upload below.",
                accept_multiple_files=accept_multiple_files,
                # on_change=self._uploader_callback,  # Disabled in forms
            )

            submit = st.form_submit_button(
                "Upload",
                type="primary" if key not in st.session_state else "secondary",
            )

        if submit:
            if uploaded_files:
                st.session_state[key] = [
                    uploaded_file.read()
                    for uploaded_file in (
                        uploaded_files if accept_multiple_files else [uploaded_files]
                    )
                ]

        if key in st.session_state:
            if len(st.session_state.get(key, [])) > 0:
                self.success(
                    f"Successfully uploaded {len(st.session_state.get(key, []))} file(s)."
                )
            else:
                self.error("No files were uploaded.")

        else:
            self.info(f"Please upload {'at least' if accept_multiple_files else ''}one file to proceed.")

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def set(self, key, value):
        self.__setattr__(key, value)

    @staticmethod
    def title(title, h=0, spacing=True):
        st.write(
            "{}{} {}".format(
                ("#" * (h if spacing is True else (6 - spacing)) + "\n") if spacing else "",
                "#" * h,
                title,
            )
        )

    @staticmethod
    def error(text="", icon="❌"):
        return st.error(f"{icon} {text}")

    @staticmethod
    def info(text="", icon="ℹ️"):
        return st.info(f"{icon} {text}")

    @staticmethod
    def success(text="", icon="✔"):  # icon="✅"
        return st.success(f"{icon} {text}")

    @staticmethod
    def warning(text="", icon="⚠️"):
        return st.warning(f"{icon} {text}")

    @staticmethod
    @st.cache_data(**CACHE_DATA)
    def cache_data(_func, *args, **kwargs):
        return _func(*args, **kwargs)

    @staticmethod
    @st.cache_data(**CACHE_TO_DISK)
    def cache_to_disk(_func, *args, **kwargs):
        return _func(*args, **kwargs)

    @staticmethod
    @st.cache_resource(**CACHE_RESOURCE)
    def cache_resource(_func, *args, **kwargs):
        return _func(*args, **kwargs)

    cache = cache_data
