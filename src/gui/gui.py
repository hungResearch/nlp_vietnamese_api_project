import streamlit as st
import requests


if 'order' not in st.session_state:
    st.session_state['order'] = []

st.set_page_config(
    page_title="Preprocessing Text",
    layout="wide",
)

st.title("Ứng dụng tiền xử lý văn bản Tiếng Việt", )

text_input = st.text_area("Nhập văn bản cần xử lý:")

option = {
    "tach_tu": "",
    "handle_encoding": ""
}

list_function = []
with st.container():
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        if st.toggle("Tách từ"):
            list_function.append("tach_tu")
            with col2:
                    option["tach_tu"] = st.selectbox(
                        "Chọn thư viện tách từ",
                        ("VnCoreNLP", "Pyvi", "Underthesea"),
                    )

with st.container():
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        if st.toggle("Xử lý encoding"):
            list_function.append("handle_encoding")
            with col2:
                    option["handle_encoding"] = st.selectbox(
                        "Chọn mã encode",
                        ("UTF-8", "Unicode", "VNI Windows", "TCVN3", "VIQR", "VPS", "VIETWARE X"),
                    )                
                
if st.toggle("Loại bỏ stop word"):
    list_function.append("stop_word")

if st.toggle("Chuẩn hóa dấu câu"):
    list_function.append("dau_cau")

if st.toggle("Chuẩn hóa dấu thanh"):
    list_function.append("dau_thanh")
                
if st.toggle("Loại bỏ mã HTML"):
    list_function.append("remove_html")
    
list_order = st.multiselect(
    "Chọn thứ tự bạn muốn xử lý:",
    list_function,
)

if st.button("Xử lý", disabled=len(list_order) != len(list_function)):
    response = requests.post("http://fastapi:8000/process", json={"text": text_input, "function": list_order, "option": option})
    result = response.json()
    st.write(f"List function: {result}")
