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
    "tach_tu": "Underthesea",
    "source_encoding": "",
    "target_encoding": ""
}

list_function = []
with st.container():
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        if st.toggle("Tách từ"):
            list_function.append("tách từ")
            with col2:
                    option["tach_tu"] = st.selectbox(
                        "Chọn thư viện tách từ",
                        ("VnCoreNLP", "Pyvi", "Underthesea"),
                    )

with st.container():
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        if st.toggle("Xử lý encoding"):
            list_function.append("xử lý encoding")
            with col2:
                    option["source_encoding"] = st.selectbox(
                        "Chọn mã encode nguồn",
                        ("UNICODE", "VNI Windows", "TCVN3", "VIQR", "VPS", "VIETWARE X"),
                    )
                    option["target_encoding"] = st.selectbox(
                        "Chọn mã encode đích",
                        ("UNICODE", "VNI Windows", "TCVN3", "VIQR", "VPS", "VIETWARE X"),
                    )
                
if st.toggle("Loại bỏ stop word"):
    list_function.append("loại bỏ hư từ")

if st.toggle("Chuẩn hóa dấu câu"):
    list_function.append("chuẩn hóa dấu câu")

if st.toggle("Chuẩn hóa dấu thanh"):
    list_function.append("chuẩn hóa dấu thanh")
                
if st.toggle("Loại bỏ mã HTML"):
    list_function.append("loại bỏ mã HTML")
    
if st.toggle("Loại bỏ khoảng trắng dư thừa"):
    list_function.append("loại bỏ khoảng trắng")
    
list_order = st.multiselect(
    "Chọn thứ tự bạn muốn xử lý:",
    list_function,
)

if st.button("Xử lý", disabled=len(list_order) != len(list_function)):
    response = requests.post("http://fastapi:8000/process", json={"text": text_input, "function": list_order, "option": option})
    result = response.json()
    st.write(result["processed_text"])
