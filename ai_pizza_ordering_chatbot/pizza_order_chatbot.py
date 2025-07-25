from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

menu="""You are OrderBot, an automated service to collect orders for a pizza restaurant. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The menu includes \
pepperoni pizza  12.95, 10.00, 7.00 \
cheese pizza   10.95, 9.25, 6.50 \
eggplant pizza   11.95, 9.75, 6.75 \
fries 4.50, 3.50 \
greek salad 7.25 \
Toppings: \
extra cheese 2.00, \
mushrooms 1.50 \
sausage 3.00 \
canadian bacon 3.50 \
AI sauce 1.50 \
peppers 1.00 \
Drinks: \
coke 3.00, 2.00, 1.00 \
sprite 3.00, 2.00, 1.00 \
bottled water 5.00 \
"""
chat_prompt = ChatPromptTemplate([
("system", menu),
MessagesPlaceholder(variable_name="chat_history"),
("user", "{query}")

])
st.header("AI Pizza Ordering System")

#store chat in a session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "AI Assistant", "content": "Hello I'm Pizza Ordering AI Assistant, How can I help you today?"}]

#history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=menu)]


#display the chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#user input
user_input = st.chat_input("Entere here")
prompt = chat_prompt.invoke({"chat_history": st.session_state.chat_history, "query": user_input})
if user_input:
    #shows user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    #LLM response
    ai_result = model.invoke(prompt)
    st.session_state.messages.append({"role": "AI Assistant", "content": ai_result.content})
    st.session_state.chat_history.append(AIMessage(content=ai_result.content))
    with st.chat_message("AI Assistant"):
        st.markdown(ai_result.content)
    