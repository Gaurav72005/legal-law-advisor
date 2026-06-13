import os
import urllib.parse
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_google_auth_url():
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")
    if not client_id:
        return None
    
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account"
    }
    return f"{auth_url}?{urllib.parse.urlencode(params)}"

def authenticate_google():
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")
    
    if not client_id or not client_secret:
        st.error("Google authentication is not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in your .env file.")
        return None

    if "code" in st.query_params:
        code = st.query_params["code"]
        
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }
        
        res = requests.post(token_url, data=data)
        if res.status_code == 200:
            access_token = res.json().get("access_token")
            user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            user_res = requests.get(user_info_url, headers={"Authorization": f"Bearer {access_token}"})
            
            if user_res.status_code == 200:
                user_info = user_res.json()
                
                # Clear the query params after successful login
                st.query_params.clear()
                
                return user_info
        else:
            st.error(f"Failed to authenticate: {res.text}")
            # Clear the query params so we don't get stuck in a loop
            st.query_params.clear()
            
    return None

def check_auth_callback():
    """
    Checks if a user is already authenticated in session state,
    or if we are returning from Google Auth with a 'code' in query params.
    """
    if "user" not in st.session_state:
        st.session_state.user = None
        
    if st.session_state.user is None and "code" in st.query_params:
        user_info = authenticate_google()
        if user_info:
            st.session_state.user = user_info
            st.rerun()

def render_login_ui():
    """
    Renders the login UI and returns True if the user is authenticated, False otherwise.
    """
    if "user" not in st.session_state:
        st.session_state.user = None
        
    # Check if returning from Google Auth
    if st.session_state.user is None and "code" in st.query_params:
        with st.spinner("Authenticating..."):
            user_info = authenticate_google()
            if user_info:
                st.session_state.user = user_info
                st.rerun()

    if st.session_state.user is None:
        st.markdown("<h2 style='text-align:center;'>Welcome to GatiNeeti</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Please sign in to continue.</p>", unsafe_allow_html=True)
        
        auth_url = get_google_auth_url()
        if auth_url:
            st.markdown(
                f'''
                <div style="display: flex; justify-content: center; margin-top: 20px;">
                    <a href="{auth_url}" target="_self" style="text-decoration: none;">
                        <button style="background-color: #ffffff; color: #757575; border: 1px solid #ddd; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: 500; display: flex; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <img src="https://www.google.com/favicon.ico" style="width: 18px; height: 18px; margin-right: 10px;">
                            Sign in with Google
                        </button>
                    </a>
                </div>
                ''', 
                unsafe_allow_html=True
            )
        return False
        
    return True

