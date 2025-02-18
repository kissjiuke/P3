import streamlit as st
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu
import bcrypt
import time
import base64
import work
def sidebar_bg(side_bg):
    side_bg_ext = 'png'

    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )
sidebar_bg('./picture/3.jpg')

def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
         }}
        header, .stToolbar {{
            visibility: hidden;
            height: 0;
        }}
        .block-container {{
            padding-top: 0 !important;
        }}
         </style>
         """,
        unsafe_allow_html=True
    )
main_bg('./picture/3.jpg')
# 初始化会话状态
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users_db" not in st.session_state:
    st.session_state.users_db = {
        "admin": bcrypt.hashpw("12345".encode('utf-8'), bcrypt.gensalt())  # 默认账号：用户名 `admin`，密码 `12345`
    }

# if "users_db" not in st.session_state:
#     st.session_state.users_db = {}  # 用于存储用户名和加密后的密码
if "show_register" not in st.session_state:
    st.session_state.show_register = False  # 控制是否显示注册界面

# 登录页面
def login():
    st.header("登录")
    st.divider()

    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")

    if st.button("登录", key="login_button"):  # 为按钮添加唯一的 key
        if username in st.session_state.users_db:
            hashed_password = st.session_state.users_db[username]
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                st.session_state.logged_in = True
                st.success("登录成功!")
                time.sleep(0.5)
                st.rerun()  # 登录成功后刷新页面
            else:
                st.error("密码错误")
        else:
            st.error("用户名不存在，请注册！")

    # 显示注册按钮
    if st.button("没有账号? 立即注册", key="show_register_button"):
        st.session_state.show_register = True  # 设置状态以显示注册界面
        st.rerun()  # 确保立即刷新页面，跳转到注册界面

# 注册页面
def register():
    st.header("注册")
    st.divider()

    new_username = st.text_input("请输入用户名")
    new_password = st.text_input("请输入密码", type="password")
    confirm_password = st.text_input("确认密码", type="password")

    if st.button("注册", key="register_button_2"):  # 为按钮添加唯一的 key
        if new_username in st.session_state.users_db:
            st.warning("用户名已存在，请选择其他用户名")
        elif new_password != confirm_password:
            st.warning("两次密码输入不一致")
        else:
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            st.session_state.users_db[new_username] = hashed_password
            st.success("注册成功！请点击返回登录按钮进行登录")

    # 显示返回登录按钮
    if st.button("已有账号? 返回登录", key="back_to_login_button"):
        st.session_state.show_register = False  # 切换回登录界面
        st.rerun()  # 确保立即刷新页面，跳转到登录界面

# 页面跳转逻辑
def main():
    if st.session_state.logged_in:
        work.all()

    else:
        if st.session_state.show_register:  # 如果显示注册界面
            register()
        else:
            login()
            # st.balloons()
# 运行主逻辑
if __name__ == "__main__":
    main()
