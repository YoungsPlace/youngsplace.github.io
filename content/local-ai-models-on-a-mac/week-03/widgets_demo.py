import operator

import streamlit as st

st.title("Hello, Streamlit!")
st.subheader("빵주의 위젯 실습")
st.markdown("**Python**으로 만든 *계산기와 할 일 목록*")

with st.sidebar:
    color = st.selectbox("색상", ["Red", "Blue", "Green"])
    count = st.slider("숫자", 1, 10, 3)
st.write(f"선택한 색상은 {color}, 숫자는 {count}입니다.")

left, right = st.columns(2)
a = left.number_input("첫 번째 수", value=12.0)
b = right.number_input("두 번째 수", value=4.0)
operations = {"+": operator.add, "-": operator.sub, "×": operator.mul, "÷": operator.truediv}
symbol = st.selectbox("연산", list(operations))
if st.button("계산"):
    if symbol == "÷" and b == 0:
        st.error("0으로 나눌 수 없습니다.")
    else:
        st.metric("결과", f"{operations[symbol](a, b):g}")

st.subheader("할 일")
if "todos" not in st.session_state:
    st.session_state.todos = []

with st.form("todo_form", clear_on_submit=True):
    todo = st.text_input("할 일 추가")
    add = st.form_submit_button("추가")
if add and todo.strip():
    st.session_state.todos.append(todo.strip())

left, right = st.columns(2)
if left.button("마지막 항목 삭제") and st.session_state.todos:
    st.session_state.todos.pop()
if right.button("전체 삭제"):
    st.session_state.todos.clear()

for item in st.session_state.todos:
    st.write("·", item)
