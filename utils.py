from __future__ import annotations
import streamlit as st
from typing import Optional

def bytes_download_button(label: str, data: bytes, filename: str, mime: str, container: Optional["st.delta_generator.DeltaGenerator"]=None):
    """
    Wrapper to place a download button either in main area or in a passed container/column.
    """
    target = container if container is not None else st
    target.download_button(label=label, data=data, file_name=filename, mime=mime)
