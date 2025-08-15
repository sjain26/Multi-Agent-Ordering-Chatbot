---
title: Multi Agent Ordering Chatbot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: mit
---

# Multi-Agent Ordering Chatbot 🤖

An intelligent chatbot system that uses multiple specialized agents to handle different types of orders efficiently.

🔗 **Live Demo**: [https://huggingface.co/spaces/jainsatyam26/intent-identifier](https://huggingface.co/spaces/jainsatyam26/intent-identifier)

## Features

- **Smart Classification**: Automatically classifies orders as generic or bulk using AI
- **Specialized Agents**: 
  - Orchestrator Agent: Initial greeting and order classification
  - Generic Order Agent: Handles standard/personal orders
  - Bulk Order Agent: Handles large quantity/wholesale orders
- **Session Management**: Tracks conversations with unique session IDs
- **Database Storage**: Saves all orders and conversations
- **Input Validation**: Comprehensive validation and error handling
- **Professional UI**: Clean Gradio interface with real-time chat

## How to Use

1. Start by describing what you want to order
2. Provide a title for your order when asked
3. Describe your order in detail
4. The system will automatically route you to the appropriate agent
5. Follow the agent's prompts to complete your order
6. View your order summary in JSON format

## Technical Details

- Built with LangChain and Groq AI
- Uses SQLite for data persistence
- Implements rate limiting and timeout handling
- Comprehensive logging for debugging
- Input sanitization for security

## Order Types

- **Generic Orders**: Personal use, small quantities (< 50 units)
- **Bulk Orders**: Large quantities, wholesale, business orders (> 50 units)

Start chatting to place your order!