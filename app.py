# Multi-Agent Ordering Chatbot System
# Requirements: pip install langchain langchain-groq gradio sqlite3 pydantic
import os

# Get API key from environment variable or use default
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
import gradio as gr
import sqlite3
import json
import uuid
import logging
import re
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import threading

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

# Configuration
DATABASE_PATH = "chatbot.db"
LOG_FILE = "chatbot.log"
MAX_INPUT_LENGTH = 500
MAX_TITLE_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 1000
API_TIMEOUT = 30  # seconds
RATE_LIMIT_CALLS = 10
RATE_LIMIT_WINDOW = 60  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    ORCHESTRATOR = "Orchestrator"
    GENERIC = "Generic"
    BULK = "Bulk"

class SessionState:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.current_agent = AgentType.ORCHESTRATOR
        self.order_data = {}
        self.conversation_history = []
        self.collecting_title = False
        self.collecting_description = False
        self.pending_handoff = None
        self.api_calls_timestamps = []  # For rate limiting
        self.error_count = 0

# Rate limiter decorator
def rate_limit(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        session_state = args[1] if len(args) > 1 else None
        if session_state:
            current_time = time.time()
            # Remove old timestamps
            session_state.api_calls_timestamps = [
                ts for ts in session_state.api_calls_timestamps 
                if current_time - ts < RATE_LIMIT_WINDOW
            ]
            
            if len(session_state.api_calls_timestamps) >= RATE_LIMIT_CALLS:
                logger.warning(f"Rate limit exceeded for session {session_state.session_id}")
                return "I'm processing too many requests. Please wait a moment before continuing.", session_state
            
            session_state.api_calls_timestamps.append(current_time)
        
        return func(self, *args, **kwargs)
    return wrapper

# Input validation functions
def sanitize_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """Sanitize and validate user input"""
    if not text or not text.strip():
        raise ValueError("Input cannot be empty")
    
    # Remove potential SQL injection characters
    text = text.replace(";", "").replace("--", "").replace("/*", "").replace("*/", "")
    
    # Trim to max length
    text = text[:max_length].strip()
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text

def validate_quantity(quantity_str: str) -> int:
    """Validate and parse quantity input"""
    try:
        # Extract numbers from string
        numbers = re.findall(r'\d+', quantity_str)
        if not numbers:
            raise ValueError("No valid number found")
        
        quantity = int(numbers[0])
        if quantity <= 0:
            raise ValueError("Quantity must be greater than 0")
        if quantity > 1000000:
            raise ValueError("Quantity seems unreasonably high. Please enter a value less than 1,000,000")
        
        return quantity
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid quantity: {str(e)}")

# Database Setup
def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create conversation table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_input TEXT,
            chatbot_response TEXT,
            agent TEXT
        )
    ''')
    
    # Create order table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "order" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            title TEXT,
            description TEXT,
            product_name TEXT,
            quantity INTEGER,
            brand_preference TEXT,
            additional_details TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_conversation(session_id: str, user_input: str, chatbot_response: str, agent: str):
    """Save conversation with error handling and logging"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversation (session_id, user_input, chatbot_response, agent)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_input, chatbot_response, agent))
        conn.commit()
        conn.close()
        logger.info(f"Conversation saved for session {session_id}, agent: {agent}")
    except sqlite3.Error as e:
        logger.error(f"Database error saving conversation: {e}")
        raise

def save_order(session_id: str, order_data: Dict[str, Any]):
    """Save order with error handling and logging"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO "order" (session_id, title, description, product_name, quantity, brand_preference, additional_details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            order_data.get('title', ''),
            order_data.get('description', ''),
            order_data.get('product_name', ''),
            order_data.get('quantity', 0),
            order_data.get('brand_preference', ''),
            json.dumps(order_data.get('additional_details', {}))
        ))
        conn.commit()
        conn.close()
        logger.info(f"Order saved for session {session_id}: {order_data.get('title', 'Untitled')}")
    except sqlite3.Error as e:
        logger.error(f"Database error saving order: {e}")
        raise

# Tools
@tool
def category_finder_tool(description: str, type_of_request: str = None) -> str:
    """
    Classifies an order request as either 'generic' or 'bulk' based on description.
    
    Args:
        description: Description of what the user wants to order
        type_of_request: Type of request (personal use or reselling)
    
    Returns:
        'generic' for single/small orders, 'bulk' for large quantity orders
    """
    # Initialize LLM for classification
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768"
    )
    
    prompt = f"""
    Analyze the following order request and classify it as either "generic" or "bulk":
    
    Description: {description}
    Type of request: {type_of_request}
    
    Classification rules:
    - "bulk" if quantity > 50 OR mentions words like "bulk", "wholesale", "mass", "large quantity", "reselling", "event", "company", "office"
    - "generic" for smaller quantities or personal use
    
    Return only "generic" or "bulk":
    """
    
    try:
        # Add timeout handling
        start_time = time.time()
        response = llm.invoke(prompt)
        
        if time.time() - start_time > API_TIMEOUT:
            logger.warning("API call timeout in category_finder_tool")
            raise TimeoutError("API call took too long")
        
        result = response.content.strip().lower()
        logger.info(f"Category classification result: {result}")
        return "bulk" if "bulk" in result else "generic"
    except Exception as e:
        logger.error(f"Error in category classification: {e}")
        # Fallback logic
        description_lower = description.lower()
        bulk_keywords = ['bulk', 'wholesale', 'mass', 'large', 'hundred', 'thousand', 'resell', 'event', 'company', 'office']
        
        # Extract numbers from description
        import re
        numbers = re.findall(r'\d+', description)
        max_number = max([int(n) for n in numbers], default=0)
        
        if max_number > 50 or any(keyword in description_lower for keyword in bulk_keywords):
            return "bulk"
        return "generic"

# Agent Classes
class OrchestratorAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
            model_name="mixtral-8x7b-32768",
            request_timeout=API_TIMEOUT
        )
        self.tools = [category_finder_tool]
    
    @rate_limit
    def process(self, user_input: str, session_state: SessionState) -> Tuple[str, SessionState]:
        try:
            # Validate input
            user_input = sanitize_input(user_input)
            if session_state.collecting_title:
                # Validate title
                title = sanitize_input(user_input, MAX_TITLE_LENGTH)
                session_state.order_data['title'] = title
                session_state.collecting_title = False
                session_state.collecting_description = True
                logger.info(f"Title collected for session {session_state.session_id}: {title}")
                return "Great! Now please describe what you want to order in detail.", session_state
        
            elif session_state.collecting_description:
                # Validate description
                description = sanitize_input(user_input, MAX_DESCRIPTION_LENGTH)
                session_state.order_data['description'] = description
                session_state.collecting_description = False
                
                # Use category finder tool with error handling
                try:
                    category = category_finder_tool.invoke({
                        "description": description,
                        "type_of_request": session_state.order_data.get('type_of_request', '')
                    })
                except Exception as e:
                    logger.error(f"Category finder error: {e}")
                    # Default to generic on error
                    category = "generic"
            
                # Hand off to appropriate agent
                if category == "bulk":
                    session_state.current_agent = AgentType.BULK
                    session_state.pending_handoff = "bulk"
                    logger.info(f"Handing off to Bulk Agent for session {session_state.session_id}")
                    return f"[Handing off to Bulk Order Agent...]\n\nI understand you need a bulk order. Let me gather the specific details for your bulk order.", session_state
                else:
                    session_state.current_agent = AgentType.GENERIC
                    session_state.pending_handoff = "generic"
                    logger.info(f"Handing off to Generic Agent for session {session_state.session_id}")
                    return f"[Handing off to Generic Order Agent...]\n\nI understand you need a standard order. Let me gather the specific details for your order.", session_state
            
            else:
                # Initial greeting and title collection
                session_state.collecting_title = True
                return "Hello! I'm here to help you with your order. Please provide a title for this order.", session_state
                
        except ValueError as e:
            logger.warning(f"Input validation error: {e}")
            return f"⚠️ {str(e)}. Please try again.", session_state
        except Exception as e:
            logger.error(f"Orchestrator processing error: {e}")
            session_state.error_count += 1
            if session_state.error_count > 3:
                return "I'm experiencing technical difficulties. Please try again later or contact support.", session_state
            return "I encountered an error. Please try again.", session_state

class GenericOrderAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
            model_name="mixtral-8x7b-32768",
            request_timeout=API_TIMEOUT
        )
        self.step = 0
    
    @rate_limit
    def process(self, user_input: str, session_state: SessionState) -> Tuple[str, SessionState]:
        try:
            # Validate input
            user_input = sanitize_input(user_input)
            if 'generic_step' not in session_state.order_data:
                session_state.order_data['generic_step'] = 0
            
            step = session_state.order_data['generic_step']
        
            # Check for handoff requests
            handoff_keywords = ['bulk', 'change', 'instead', 'actually', 'different']
            if any(keyword in user_input.lower() for keyword in handoff_keywords) and step > 0:
                # Parse the new request
                session_state.current_agent = AgentType.ORCHESTRATOR
                session_state.order_data = {}
                session_state.collecting_title = True
                logger.info(f"Generic agent handing back to orchestrator for session {session_state.session_id}")
                return "I understand you want to change your order. Please provide a new title for your order.", session_state
        
            if step == 0:
                # Ask for product name
                session_state.order_data['generic_step'] = 1
                return "What is the specific product name you want to order?", session_state
            
            elif step == 1:
                # Collect product name, ask for quantity
                product_name = sanitize_input(user_input, MAX_INPUT_LENGTH)
                session_state.order_data['product_name'] = product_name
                session_state.order_data['generic_step'] = 2
                logger.info(f"Product name collected: {product_name}")
                return "How many units do you need?", session_state
        
            elif step == 2:
                # Collect quantity, ask for brand preference
                try:
                    quantity = validate_quantity(user_input)
                    session_state.order_data['quantity'] = quantity
                    session_state.order_data['generic_step'] = 3
                    logger.info(f"Quantity collected: {quantity}")
                    return "Do you have any brand preference? (Enter 'No' if none)", session_state
                except ValueError as e:
                    logger.warning(f"Invalid quantity input: {user_input}")
                    return f"⚠️ {str(e)}. Please enter a valid number.", session_state
        
            elif step == 3:
                # Collect brand preference and finalize
                brand_pref = sanitize_input(user_input, MAX_INPUT_LENGTH) if user_input.lower() not in ['no', 'none', 'n/a'] else ''
                session_state.order_data['brand_preference'] = brand_pref
                
                try:
                    # Save to database
                    save_order(session_state.session_id, session_state.order_data)
                    
                    # Generate summary
                    summary = self.generate_summary(session_state.order_data)
                    
                    # Reset for next order
                    session_state.current_agent = AgentType.ORCHESTRATOR
                    session_state.order_data = {}
                    session_state.error_count = 0  # Reset error count on success
                    
                    logger.info(f"Generic order completed for session {session_state.session_id}")
                    return f"Perfect! Here's your order summary:\n\n{summary}\n\nOrder saved successfully! How can I help you with another order?", session_state
                except Exception as e:
                    logger.error(f"Error saving order: {e}")
                    return "There was an error saving your order. Please try again or contact support.", session_state
                    
        except ValueError as e:
            logger.warning(f"Input validation error in generic agent: {e}")
            return f"⚠️ {str(e)}. Please try again.", session_state
        except Exception as e:
            logger.error(f"Generic agent processing error: {e}")
            session_state.error_count += 1
            if session_state.error_count > 3:
                return "I'm experiencing technical difficulties. Please try again later or contact support.", session_state
            return "I encountered an error. Please try again.", session_state
    
    def generate_summary(self, order_data: Dict[str, Any]) -> str:
        return f"""📋 **ORDER SUMMARY**
**Title:** {order_data.get('title', 'N/A')}
**Description:** {order_data.get('description', 'N/A')}
**Product:** {order_data.get('product_name', 'N/A')}
**Quantity:** {order_data.get('quantity', 'N/A')} units
**Brand Preference:** {order_data.get('brand_preference', 'No preference')}

JSON Format:
```json
{json.dumps({
    "title": order_data.get('title', ''),
    "description": order_data.get('description', ''),
    "product_name": order_data.get('product_name', ''),
    "quantity": order_data.get('quantity', 0),
    "brand_preference": order_data.get('brand_preference', '')
}, indent=2)}
```"""

class BulkOrderAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
            model_name="mixtral-8x7b-32768",
            request_timeout=API_TIMEOUT
        )
    
    @rate_limit
    def process(self, user_input: str, session_state: SessionState) -> Tuple[str, SessionState]:
        try:
            # Validate input
            user_input = sanitize_input(user_input)
            if 'bulk_step' not in session_state.order_data:
                session_state.order_data['bulk_step'] = 0
            
            step = session_state.order_data['bulk_step']
        
            # Check for handoff requests
            handoff_keywords = ['change', 'instead', 'actually', 'different']
            if any(keyword in user_input.lower() for keyword in handoff_keywords) and step > 0:
                # Parse the new request
                session_state.current_agent = AgentType.ORCHESTRATOR
                session_state.order_data = {}
                session_state.collecting_title = True
                logger.info(f"Bulk agent handing back to orchestrator for session {session_state.session_id}")
                return "I understand you want to change your order. Please provide a new title for your order.", session_state
        
            if step == 0:
                # Ask for product type
                session_state.order_data['bulk_step'] = 1
                return "What type of products are you ordering in bulk?", session_state
            
            elif step == 1:
                # Collect product type, ask for quantity
                product_type = sanitize_input(user_input, MAX_INPUT_LENGTH)
                session_state.order_data['product_name'] = product_type
                session_state.order_data['bulk_step'] = 2
                logger.info(f"Bulk product type collected: {product_type}")
                return "What is the total quantity or units needed?", session_state
        
            elif step == 2:
                # Collect quantity, ask for supplier preference
                try:
                    quantity = validate_quantity(user_input)
                    # Additional check for bulk orders
                    if quantity < 50:
                        logger.warning(f"Bulk order with small quantity: {quantity}")
                    session_state.order_data['quantity'] = quantity
                    session_state.order_data['bulk_step'] = 3
                    logger.info(f"Bulk quantity collected: {quantity}")
                    return "Do you have any supplier preference or constraints? (Enter 'No' if none)", session_state
                except ValueError as e:
                    logger.warning(f"Invalid bulk quantity input: {user_input}")
                    return f"⚠️ {str(e)}. Please enter a valid number.", session_state
        
            elif step == 3:
                # Collect supplier preference and finalize
                supplier_pref = sanitize_input(user_input, MAX_INPUT_LENGTH) if user_input.lower() not in ['no', 'none', 'n/a'] else ''
                session_state.order_data['brand_preference'] = supplier_pref
                
                try:
                    # Save to database
                    save_order(session_state.session_id, session_state.order_data)
                    
                    # Generate summary
                    summary = self.generate_summary(session_state.order_data)
                    
                    # Reset for next order
                    session_state.current_agent = AgentType.ORCHESTRATOR
                    session_state.order_data = {}
                    session_state.error_count = 0  # Reset error count on success
                    
                    logger.info(f"Bulk order completed for session {session_state.session_id}")
                    return f"Excellent! Here's your bulk order summary:\n\n{summary}\n\nBulk order saved successfully! How can I help you with another order?", session_state
                except Exception as e:
                    logger.error(f"Error saving bulk order: {e}")
                    return "There was an error saving your bulk order. Please try again or contact support.", session_state
                    
        except ValueError as e:
            logger.warning(f"Input validation error in bulk agent: {e}")
            return f"⚠️ {str(e)}. Please try again.", session_state
        except Exception as e:
            logger.error(f"Bulk agent processing error: {e}")
            session_state.error_count += 1
            if session_state.error_count > 3:
                return "I'm experiencing technical difficulties. Please try again later or contact support.", session_state
            return "I encountered an error. Please try again.", session_state
    
    def generate_summary(self, order_data: Dict[str, Any]) -> str:
        return f"""📋 **BULK ORDER SUMMARY**
**Title:** {order_data.get('title', 'N/A')}
**Description:** {order_data.get('description', 'N/A')}
**Product Type:** {order_data.get('product_name', 'N/A')}
**Total Quantity:** {order_data.get('quantity', 'N/A')} units
**Supplier Preference:** {order_data.get('brand_preference', 'No preference')}

JSON Format:
```json
{json.dumps({
    "title": order_data.get('title', ''),
    "description": order_data.get('description', ''),
    "product_name": order_data.get('product_name', ''),
    "quantity": order_data.get('quantity', 0),
    "brand_preference": order_data.get('brand_preference', '')
}, indent=2)}
```"""

# Main Chatbot System
class MultiAgentChatbot:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.generic_agent = GenericOrderAgent()
        self.bulk_agent = BulkOrderAgent()
        self.session_states = {}
    
    def get_session_state(self, session_id: str) -> SessionState:
        if session_id not in self.session_states:
            self.session_states[session_id] = SessionState()
        return self.session_states[session_id]
    
    def process_message(self, message: str, session_id: str) -> str:
        try:
            session_state = self.get_session_state(session_id)
            
            # Log incoming message
            logger.info(f"Processing message for session {session_id}: {message[:50]}...")
            
            # Route to appropriate agent
            if session_state.current_agent == AgentType.ORCHESTRATOR:
                response, session_state = self.orchestrator.process(message, session_state)
                agent_name = "Orchestrator"
            elif session_state.current_agent == AgentType.GENERIC:
                response, session_state = self.generic_agent.process(message, session_state)
                agent_name = "Generic"
            elif session_state.current_agent == AgentType.BULK:
                response, session_state = self.bulk_agent.process(message, session_state)
                agent_name = "Bulk"
            
            # Update session state
            self.session_states[session_id] = session_state
            
            # Save conversation to database
            try:
                save_conversation(session_state.session_id, message, response, agent_name)
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
                # Continue processing even if save fails
            
            # Add agent indicator to response
            agent_indicator = f"[Running {agent_name} Agent]\n" if agent_name != "Orchestrator" or session_state.pending_handoff else ""
            
            return f"{agent_indicator}{response}"
            
        except Exception as e:
            logger.error(f"Critical error in process_message: {e}", exc_info=True)
            return "I'm sorry, I encountered an unexpected error. Please try again or contact support."

# Initialize system
init_database()
chatbot = MultiAgentChatbot()

# Gradio Interface
def chat_interface(message, history, session_id):
    try:
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Validate message is not empty
        if not message or not message.strip():
            return history, "", session_id
        
        response = chatbot.process_message(message, session_id)
        # For messages format, append as dictionaries
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        return history, "", session_id
    except Exception as e:
        logger.error(f"Error in chat interface: {e}")
        error_response = "I'm sorry, there was an error processing your message. Please try again."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_response})
        return history, "", session_id

def reset_chat():
    new_session_id = str(uuid.uuid4())
    logger.info(f"Chat reset, new session: {new_session_id}")
    return [], "", new_session_id

# Create Gradio interface
with gr.Blocks(title="Multi-Agent Ordering Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Multi-Agent Ordering Chatbot
    
    This chatbot helps you place orders efficiently by routing you to specialized agents:
    - **Orchestrator Agent**: Initial classification and routing
    - **Generic Order Agent**: Handles standard/personal orders
    - **Bulk Order Agent**: Handles large quantity/wholesale orders
    
    Start by describing what you want to order!
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot_interface = gr.Chatbot(
                label="Chat with the Ordering Assistant",
                height=500,
                show_copy_button=True,
                type="messages"
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Your message",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Reset Chat", variant="secondary")
        
        with gr.Column(scale=1):
            session_state_display = gr.Textbox(
                label="Session ID",
                value=str(uuid.uuid4()),
                interactive=False
            )
    
    # Event handlers
    submit_btn.click(
        chat_interface,
        inputs=[user_input, chatbot_interface, session_state_display],
        outputs=[chatbot_interface, user_input, session_state_display]
    )
    
    user_input.submit(
        chat_interface,
        inputs=[user_input, chatbot_interface, session_state_display],
        outputs=[chatbot_interface, user_input, session_state_display]
    )
    
    clear_btn.click(
        reset_chat,
        outputs=[chatbot_interface, user_input, session_state_display]
    )

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting Multi-Agent Ordering Chatbot...")
    print("📝 Make sure to set your GROQ_API_KEY in the code!")
    print(f"📂 Logs will be saved to: {LOG_FILE}")
    logger.info("Application started")
    
    try:
        # For Hugging Face Spaces, let it handle port allocation
        demo.launch()
    except Exception as e:
        logger.critical(f"Failed to launch application: {e}")
        raise
