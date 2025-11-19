# # ui/login.py
# import streamlit as st
# from models.database import get_database

# def show_login_page():
#     """Display login/register page"""
    
#     st.markdown("""
#     <div style='text-align:center; margin-top:30px;'>
#         <img src='https://img.icons8.com/fluency/96/artificial-intelligence.png' width='100'/>
#         <h1 style='color:#00b894; font-size:3rem; margin-top:20px;'>ProTrader AI</h1>
#         <p style='color:#888; font-size:1.2rem;'>Multi-Agent Trading Platform</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     # Toggle between login and register
#     if 'auth_mode' not in st.session_state:
#         st.session_state.auth_mode = 'login'
    
#     col1, col2, col3 = st.columns([1, 2, 1])
    
#     with col2:
#         # Mode toggle buttons
#         tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
#         with tab1:
#             show_login_form()
        
#         with tab2:
#             show_register_form()


# def show_login_form():
#     """Display login form"""
#     st.markdown("### Welcome Back!")
#     st.markdown("Login to access your portfolio and start trading")
    
#     with st.form("login_form"):
#         username = st.text_input("Username", placeholder="Enter your username")
#         password = st.text_input("Password", type="password", placeholder="Enter your password")
        
#         submit = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
        
#         if submit:
#             if not username or not password:
#                 st.error("⚠️ Please fill in all fields")
#             else:
#                 db = get_database()
#                 success, user_id, message = db.login_user(username, password)
                
#                 if success:
#                     st.session_state.logged_in = True
#                     st.session_state.user_id = user_id
#                     st.session_state.username = username
#                     st.session_state.page = 'home'
#                     st.success(message)
#                     st.balloons()
#                     st.rerun()
#                 else:
#                     st.error(message)


# def show_register_form():
#     """Display registration form"""
#     st.markdown("### Create Your Account")
#     st.markdown("Register to get ₹10,00,000 virtual capital and start trading!")
    
#     with st.form("register_form"):
#         username = st.text_input("Username", placeholder="Choose a unique username")
#         email = st.text_input("Email", placeholder="your.email@example.com")
#         password = st.text_input("Password", type="password", placeholder="Create a strong password")
#         confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        
#         agree = st.checkbox("I agree that this is for educational purposes only (paper trading)")
        
#         submit = st.form_submit_button("📝 Create Account", use_container_width=True, type="primary")
        
#         if submit:
#             if not username or not email or not password or not confirm_password:
#                 st.error("⚠️ Please fill in all fields")
#             elif password != confirm_password:
#                 st.error("⚠️ Passwords do not match")
#             elif len(password) < 6:
#                 st.error("⚠️ Password must be at least 6 characters long")
#             elif not agree:
#                 st.error("⚠️ Please agree to the terms")
#             else:
#                 db = get_database()
#                 success, message = db.register_user(username, email, password)
                
#                 if success:
#                     st.success(message)
#                     st.balloons()
#                     st.info("Please login with your credentials")
#                 else:
#                     st.error(message)
# ui/login.py
import streamlit as st
from models.database import get_database

def show_login_page():
    """Display login/register page"""
    
    st.markdown("""
    <div style='text-align:center; margin-top:30px;'>
        <img src='https://img.icons8.com/fluency/96/artificial-intelligence.png' width='100'/>
        <h1 style='color:#00b894; font-size:3rem; margin-top:20px;'>ProTrader AI</h1>
        <p style='color:#888; font-size:1.2rem;'>Multi-Agent Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Toggle between login and register
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Mode toggle buttons
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            show_login_form()
        
        with tab2:
            show_register_form()


def show_login_form():
    """Display login form"""
    st.markdown("### Welcome Back!")
    st.markdown("Login to access your portfolio and start trading")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submit = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
        
        if submit:
            if not username or not password:
                st.error("⚠️ Please fill in all fields")
            else:
                db = get_database()
                success, user_id, message = db.login_user(username, password)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    
                    # Check if user is admin
                    st.session_state.is_admin = db.is_admin(username)
                    
                    st.session_state.page = 'home'
                    st.success(message)
                    
                    if st.session_state.is_admin:
                        st.success("🔑 Admin privileges activated!")
                    
                    st.balloons()
                    st.rerun()
                else:
                    st.error(message)


def show_register_form():
    """Display registration form"""
    st.markdown("### Create Your Account")
    st.markdown("Register to get ₹10,00,000 virtual capital and start trading!")
    
    with st.form("register_form"):
        username = st.text_input("Username", placeholder="Choose a unique username")
        email = st.text_input("Email", placeholder="your.email@example.com")
        password = st.text_input("Password", type="password", placeholder="Create a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        
        agree = st.checkbox("I agree that this is for educational purposes only (paper trading)")
        
        submit = st.form_submit_button("📝 Create Account", use_container_width=True, type="primary")
        
        if submit:
            if not username or not email or not password or not confirm_password:
                st.error("⚠️ Please fill in all fields")
            elif password != confirm_password:
                st.error("⚠️ Passwords do not match")
            elif len(password) < 6:
                st.error("⚠️ Password must be at least 6 characters long")
            elif not agree:
                st.error("⚠️ Please agree to the terms")
            else:
                db = get_database()
                success, message = db.register_user(username, email, password)
                
                if success:
                    st.success(message)
                    st.balloons()
                    st.info("✅ Please login with your credentials")
                else:
                    st.error(message)
