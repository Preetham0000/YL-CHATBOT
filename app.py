import streamlit as st
import re
from pathlib import Path
from yantra_rag.config import load_settings
from yantra_rag.rag_agent import YantraRAGAgent

# Page config
st.set_page_config(
    page_title="Yantra Live RAG",
    page_icon="assets/logo.jpg",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_agent():
    settings = load_settings()
    return YantraRAGAgent(settings)

def display_images_for_page(page_num: int):
    """Display images for a specific page number."""
    # Define the base path for images
    # Hardcoded for the JCB manual as per current requirement
    image_dir = Path("data/processed/images/JCB_Service manual 3CX 4CX")
    if not image_dir.exists():
        return

    # Find images for this page
    # Pattern: *_page{page_num}_img*.* OR *_page{page_num}_render*.*
    images = []
    for ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
        images.extend(list(image_dir.glob(f"*_page{page_num}_img*.{ext}")))
        images.extend(list(image_dir.glob(f"*_page{page_num}_render*.{ext}")))
    
    # Sort images to ensure consistent order
    images.sort()
    
    if images:
        st.markdown(f"**Images from Page {page_num}:**")
        # Display in a grid
        cols = st.columns(min(len(images), 3)) if len(images) > 0 else [st]
        for idx, img_path in enumerate(images):
            with cols[idx % len(cols)]:
                st.image(str(img_path), caption=f"Page {page_num} Image {idx+1}", use_container_width=True)

def main():
    st.title("Yantra Live Support Agent")

    # Sidebar for settings or info
    with st.sidebar:
        if Path("assets/logo.jpg").exists():
            st.image("assets/logo.jpg", use_container_width=True)
        
        st.header("About")
        st.markdown("""
        This AI assistant answers questions about Yantra Live products using the provided knowledge base.
        
        **Capabilities:**
        - Technical specifications
        - Compatibility checks
        - Model comparisons
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Load agent
    try:
        agent = get_agent()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Check for page tags and display images for assistant messages
            if message["role"] == "assistant":
                # Match [PAGE: 123], PAGE 123, Page 123
                page_matches = re.findall(r"(?:\[PAGE:|PAGE)\s*(\d+)(?:\])?", message["content"], re.IGNORECASE)
                # Deduplicate page numbers
                unique_pages = sorted(list(set(int(p) for p in page_matches)))
                for page_num in unique_pages:
                    display_images_for_page(page_num)
            
            # Display suggestions if available
            if message.get("suggestions"):
                st.markdown("---")
                st.caption("Suggested Follow-up Questions:")
                cols = st.columns(2)
                for j, question in enumerate(message["suggestions"]):
                    if cols[j % 2].button(question, key=f"suggestion_{i}_{j}"):
                        st.session_state.clicked_suggestion = question
                        st.rerun()

    # Handle input from chat_input or clicked suggestion
    prompt = st.chat_input("Ask a question about machines, parts, or compatibility...")
    
    # Override prompt if a suggestion was clicked
    if "clicked_suggestion" in st.session_state:
        prompt = st.session_state.clicked_suggestion
        del st.session_state.clicked_suggestion

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message (if not already displayed by rerun, but we appended it so it will be displayed next run? 
        # No, we are in the same run. We need to display it now or just let the next rerun handle it?
        # Standard Streamlit pattern is to append and then rerun or display immediately.
        # Since we are at the bottom, we can display it.)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Pass conversation history (excluding the latest user message which is 'prompt')
                    history = st.session_state.messages[:-1]
                    response = agent.answer(prompt, history=history)
                    st.markdown(response.answer)
                    
                    # Add assistant response to chat history
                    msg_data = {
                        "role": "assistant", 
                        "content": response.answer
                    }
                    if response.suggested_questions:
                        msg_data["suggestions"] = response.suggested_questions
                    
                    st.session_state.messages.append(msg_data)
                    
                    # Force rerun to show the new message and suggestions properly
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
