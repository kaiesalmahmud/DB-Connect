import os
import openai
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

import streamlit as st

API_KEY = open('key.txt', 'r').read().strip()
DB_PASSWORD = open('pass.txt', 'r').read().strip()
os.environ["OPENAI_API_KEY"] = API_KEY

openai.api_key = API_KEY

from dotenv import load_dotenv
load_dotenv()


SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Following are the unique values in some of the columns of the database:

Unique values in column 'cost_category': [nan 'Operating Expense' 'Field Asset' 'Administrative Expense'
 'Fixed Asset' 'Receipt']


Unique values in column 'cost_subcategory': ['Balance B/D' 'Entertainment (Admin)' 'CEO' 'City Group Accounts'
 'Hasan & Brothers' 'Electric Equipment' 'Office Stationary'
 'Computers & Printers' 'Pantry Supplies' 'IOU' 'Entertainment (Ops)'
 'Travelling & Conveyance' 'ISP' 'Medicine' 'Carrying Cost'
 'Sheba.xyz- digiGO' 'Stata IT Limited' 'Retail Partner'
 'Software & Subscription Fees' 'Electric Tools - Tv Installation'
 'Router' 'Bkash' 'Salary (Op)' 'Sales' 'Bill Reimbursement'
 'Final Settlement' 'Office Decoration/Reconstruction'
 'Office Refreshment' 'Advertising' 'Festival Bonus'
 'Deployment Equipments ' 'Misc. Exp' nan 'Furniture & Fixtures'
 'Software & Subscriptions' 'Electrician - Tv Installation'
 'Lunch Allowance' 'Training & Development' 'KPI Bonus' 'Office Equipment']


Unique values in column 'Holder/Bearer/Vendor': [nan 'Staffs' 'CEO' 'Accounts (AD-IQ)' 'Vendor' 'Morshed' 'SR' 'WC'
 'Tea Spices' 'Salim' 'Amzad' 'Masud' 'ISP' 'Shoikot' 'Shahin' 'Rakibul'
 'Rubab' 'Retail Partner' 'Asif' 'Aman' 'A/C Payable' 'H & H Construction'
 'Printer' '32" Tv' 'Router' 'Android Box Tx6' 'Tv Frame'
 'Electric Spare Tools' 'Tonner Cartridge' 'Digi Jadoo Broadband Ltd'
 'Shamim' 'Labib' 'Teamviewer' 'Eid-ul-fitre' 'Omran' 'Hasan & Brothers'
 'Flat flexible cable' 'Umbrella' 'Flash Net Enterprise' 'April'
 'Working Capital' 'Driver' 'Condensed Milk' '100' '75'
 "Retail Partner's Payment" 'Grid Ventures Ltd' 'Nut+Boltu' 'Sugar' 'Tea'
 'Coffee' 'Coffee Mate' '25' 'SSD 256 GB' 'Electrician' 'May' 'Emon' 'Jun'
 'Farib & Indec']


Unique values in column 'Source': ['50K' 'SR' nan]

"""

SQL_FUNCTIONS_SUFFIX = """I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables."""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""


def get_response(input_text):
    response = agent_executor(input_text)

    # print(response['intermediate_steps'][1][0].tool)
    # print(response['intermediate_steps'][-1][0].tool)
    # print(response['output'])


    if response['intermediate_steps'][1][0].tool == 'sql_db_schema':
        schema = response['intermediate_steps'][1][1]
    else: schema = None

    if response['intermediate_steps'][-1][0].tool == 'sql_db_query':
        query = response['intermediate_steps'][-1][0].tool_input
        query_output = response['intermediate_steps'][-1][1]
    else: query, query_output = None, None

    answer = response['output']

    return schema, query, query_output, answer

def explain(query, schema, query_output):

    message_history = [{"role": "user", "content": f"""You are a SQL query explainer bot. That means you will explain the logic of a SQL query. 
                    There is a postgreSQL database table with the following table:

                    {schema}                   
                    
                    A SQL query is executed on the table and it returns the following result:

                    {query_output}

                    I will give you the SQL query executed to get the result and you will explain the logic executed in the query.
                    Make the explanation brief and simple. It will be used as the explanation of the results. Do not mention the query itself.
                    No need to explain the total query. Just explain the logic of the query.
                    Reply only with the explaination to further input. If you understand, say OK."""},
                   {"role": "assistant", "content": f"OK"}]

    message_history.append({"role": "user", "content": query})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    explaination = completion.choices[0].message.content
    return explaination


host="ep-wispy-forest-393400.ap-southeast-1.aws.neon.tech"
port="5432"
database="accountsDB"
username="db_user"
password=DB_PASSWORD

# Create the sidebar for DB connection parameters
st.sidebar.header("Connect Your Database")
host = st.sidebar.text_input("Host", value=host)
port = st.sidebar.text_input("Port", value=port)
username = st.sidebar.text_input("Username", value=username)
password = st.sidebar.text_input("Password", value=password)
database = st.sidebar.text_input("Database", value=database)
# submit_button = st.sidebar.checkbox("Connect")

db = SQLDatabase.from_uri(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=SQL_PREFIX,
    suffix=SQL_FUNCTIONS_SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    agent_executor_kwargs = {'return_intermediate_steps': True}
)

# Create the main panel
st.title("DB Connect :cyclone:")
st.subheader("You are connected to AD-IQ Accounts database!!")
st.caption("The database contains the Daily Cash Input Output data for AD-IQ Accounts from Jan to June")


st.divider()
st.write("*--Helpful Info--*")
st.text("""
Cost categories: 
'Operating Expense' 'Field Asset' 'Administrative Expense'
'Fixed Asset' 'Receipt'


Cost Subcategories: 
'Balance B/D' 'Entertainment (Admin)' 'CEO' 'City Group Accounts'
'Hasan & Brothers' 'Electric Equipment' 'Office Stationary'
'Computers & Printers' 'Pantry Supplies' 'IOU' 'Entertainment (Ops)'
'Travelling & Conveyance' 'ISP' 'Medicine' 'Carrying Cost'
'Sheba.xyz- digiGO' 'Stata IT Limited' 'Retail Partner'
'Software & Subscription Fees' 'Electric Tools - Tv Installation'
'Router' 'Bkash' 'Salary (Op)' 'Sales' 'Bill Reimbursement'
'Final Settlement' 'Office Decoration/Reconstruction'
'Office Refreshment' 'Advertising' 'Festival Bonus'
'Deployment Equipments ' 'Misc. Exp' nan 'Furniture & Fixtures'
'Software & Subscriptions' 'Electrician - Tv Installation'
'Lunch Allowance' 'Training & Development' 'KPI Bonus' 'Office Equipment'


List of Holder/Bearer/Vendor: 
'Staffs' 'CEO' 'Accounts (AD-IQ)' 'Vendor' 'Morshed' 'SR' 'WC'
'Tea Spices' 'Salim' 'Amzad' 'Masud' 'ISP' 'Shoikot' 'Shahin' 'Rakibul'
'Rubab' 'Retail Partner' 'Asif' 'Aman' 'A/C Payable' 'H & H Construction'
'Printer' '32" Tv' 'Router' 'Android Box Tx6' 'Tv Frame'
'Electric Spare Tools' 'Tonner Cartridge' 'Digi Jadoo Broadband Ltd'
'Shamim' 'Labib' 'Teamviewer' 'Eid-ul-fitre' 'Omran' 'Hasan & Brothers'
'Flat flexible cable' 'Umbrella' 'Flash Net Enterprise' 'April'
'Working Capital' 'Driver' 'Condensed Milk' '100' '75'
"Retail Partner's Payment" 'Grid Ventures Ltd' 'Nut+Boltu' 'Sugar' 'Tea'
'Coffee' 'Coffee Mate' '25' 'SSD 256 GB' 'Electrician' 'May' 'Emon' 'Jun'
'Farib & Indec'
""")
st.divider()

# Get the user's natural question input
question = st.text_input(":blue[Ask a question:]", placeholder="Enter your question.")

# Create a submit button for executing the query
query_button = st.button("Submit")

# Execute the query when the submit button is clicked
if query_button:

    # Display the results as a dataframe
    # Execute the query and get the results as a dataframe
    try:
        with st.spinner('Calculating...'):
            print("\nQuestion: " + str(question))
            # print(str(question))
            schema, query, query_output, answer = get_response(question)

            if query:
                explaination = explain(query, schema, query_output)
            else: explaination = None

            # explaination = explain(query, schema, query_output)

        # if query:
        #     print("\nExplaination: " + str(explaination))

        print("\nExplaination: " + str(explaination))

        st.subheader("Answer :robot_face:")
        st.write(answer)

        try:
            if query:

                # st.caption("Query:")
                # st.caption(query)
                st.divider()

                st.caption("Explaination:")
                st.caption(explaination)

                st.divider()
        except Exception as e:
            print(e)

        st.info(":coffee: _Did that answer your question? If not, try to be more specific._")
    except:
        st.warning(":wave: Please enter a valid question. Try to be as specific as possible.")
