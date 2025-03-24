<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Enable AI to control your browser 🤖</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

🌐 Browser-use is the easiest way to connect your AI agents with the browser.

💡 See what others are building and share your projects in our [Discord](https://link.browser-use.com/discord)! Want Swag? Check out our [Merch store](https://browsermerch.com).

🌤️ Skip the setup - try our <b>hosted version</b> for instant browser automation! <b>[Try the cloud ☁︎](https://cloud.browser-use.com)</b>.

# Quick start

With pip (Python>=3.11):

```bash
pip install browser-use
```

Install Playwright: 
```bash
playwright install chromium
```

Spin up your agent:

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
load_dotenv()

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    await agent.run()

asyncio.run(main())
```

Add your API keys for the provider you want to use to your `.env` file.

```bash
OPENAI_API_KEY=
```

For other settings, models, and more, check out the [documentation 📕](https://docs.browser-use.com).

### Test with UI

You can test [browser-use with a UI repository](https://github.com/browser-use/web-ui)

Or simply run the gradio example:

```
uv pip install gradio
```

```bash
python examples/ui/gradio_demo.py
```

# Demos

<br/><br/>

[Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py): Add grocery items to cart, and checkout.

[![AI Did My Groceries](https://github.com/user-attachments/assets/d9359085-bde6-41d4-aa4e-6520d0221872)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

Prompt: Add my latest LinkedIn follower to my leads in Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/1440affc-a552-442e-b702-d0d3b277b0ae)

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.'

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py): Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py): Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/><br/>

## More examples

For more examples see the [examples](examples) folder or join the [Discord](https://link.browser-use.com/discord) and show off your project.

# Vision

Tell your computer what to do, and it gets it done.

## Roadmap

### Agent

- [ ] Improve agent memory (summarize, compress, RAG, etc.)
- [ ] Enhance planning capabilities (load website specific context)
- [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

- [ ] Improve extraction for datepickers, dropdowns, special elements
- [ ] Improve state representation for UI elements

### Rerunning tasks

- [ ] LLM as fallback
- [ ] Make it easy to define workfows templates where LLM fills in the details
- [ ] Return playwright script from the agent

### Datasets

- [ ] Create datasets for complex tasks
- [ ] Benchmark various models against each other
- [ ] Fine-tuning models for specific tasks

### User Experience

- [ ] Human-in-the-loop execution
- [ ] Improve the generated GIF quality
- [ ] Create various demos for tutorial execution, job application, QA testing, social media, etc.

## Contributing

We love contributions! Feel free to open issues for bugs or feature requests. To contribute to the docs, check out the `/docs` folder.

## Local Setup

To learn more about the library, check out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

## Cooperations

We are forming a commission to define best practices for UI/UX design for browser agents.
Together, we're exploring how software redesign improves the performance of AI agents and gives these companies a competitive advantage by designing their existing software to be at the forefront of the agent age.

Email [Toby](mailto:tbiddle@loop11.com?subject=I%20want%20to%20join%20the%20UI/UX%20commission%20for%20AI%20agents&body=Hi%20Toby%2C%0A%0AI%20found%20you%20in%20the%20browser-use%20GitHub%20README.%0A%0A) to apply for a seat on the committee.

## Swag

Want to show off your Browser-use swag? Check out our [Merch store](https://browsermerch.com). Good contributors will receive swag for free 👀.

## Citation

If you use Browser Use in your research or project, please cite:

```bibtex
@software{browser_use2024,
  author = {Müller, Magnus and Žunič, Gregor},
  title = {Browser Use: Enable AI to control your browser},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/browser-use/browser-use}
}
```

 <div align="center"> <img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400"/> 

 
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/mamagnus00)
 
 </div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
 </div>

 ---

 Get Started
Quickstart
Start using Browser Use with this quickstart guide

​
Prepare the environment
Browser Use requires Python 3.11 or higher.

First, we recommend using uv to setup the Python environment.


Copy
uv venv --python 3.11
and activate it with:


Copy
# For Mac/Linux:
source .venv/bin/activate

# For Windows:
.venv\Scripts\activate
Install the dependencies:


Copy
uv pip install browser-use
Then install playwright:


Copy
uv run playwright install
​
Create an agent
Then you can use the agent as follows:

agent.py

Copy
from langchain_openai import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
load_dotenv()

import asyncio

llm = ChatOpenAI(model="gpt-4o")

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())
​
Set up your LLM API keys
ChatOpenAI and other Langchain chat models require API keys. You should store these in your .env file. For example, for OpenAI and Anthropic, you can set the API keys in your .env file, such as:

.env

Copy
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
For other LLM models you can refer to the Langchain documentation to find how to set them up with their specific API keys.

---

Customize
Supported Models
Guide to using different LangChain chat models with Browser Use

​
Overview
Browser Use supports various LangChain chat models. Here’s how to configure and use the most popular ones. The full list is available in the LangChain documentation.

​
Model Recommendations
We have yet to test performance across all models. Currently, we achieve the best results using GPT-4o with an 89% accuracy on the WebVoyager Dataset. DeepSeek-V3 is 30 times cheaper than GPT-4o. Gemini-2.0-exp is also gaining popularity in the community because it is currently free. We also support local models, like Qwen 2.5, but be aware that small models often return the wrong output structure-which lead to parsing errors. We believe that local models will improve significantly this year.

All models require their respective API keys. Make sure to set them in your environment variables before running the agent.

​
Supported Models
All LangChain chat models, which support tool-calling are available. We will document the most popular ones here.

​
OpenAI
OpenAI’s GPT-4o models are recommended for best performance.


Copy
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)
Required environment variables:

.env

Copy
OPENAI_API_KEY=
​
Anthropic

Copy
from langchain_anthropic import ChatAnthropic
from browser_use import Agent

# Initialize the model
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.0,
    timeout=100, # Increase for complex tasks
)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)
And add the variable:

.env

Copy
ANTHROPIC_API_KEY=
​
Azure OpenAI

Copy
from langchain_openai import AzureChatOpenAI
from browser_use import Agent
from pydantic import SecretStr
import os

# Initialize the model
llm = AzureChatOpenAI(
    model="gpt-4o",
    api_version='2024-10-21',
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
    api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)
Required environment variables:

.env

Copy
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=
​
Gemini

Copy
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from pydantic import SecretStr
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(os.getenv('GEMINI_API_KEY')))

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)
Required environment variables:

.env

Copy
GEMINI_API_KEY=
​
DeepSeek-V3
The community likes DeepSeek-V3 for its low price, no rate limits, open-source nature, and good performance. The example is available here.


Copy
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr


# Initialize the model
llm=ChatOpenAI(base_url='https://api.deepseek.com/v1', model='deepseek-chat', api_key=SecretStr(api_key))

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm,
    use_vision=False
)
Required environment variables:

.env

Copy
DEEPSEEK_API_KEY=
​
DeepSeek-R1
We support DeepSeek-R1. Its not fully tested yet, more and more functionality will be added, like e.g. the output of it’sreasoning content. The example is available here. It does not support vision. The model is open-source so you could also use it with Ollama, but we have not tested it.


Copy
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr


# Initialize the model
llm=ChatOpenAI(base_url='https://api.deepseek.com/v1', model='deepseek-reasoner', api_key=SecretStr(api_key))

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm,
    use_vision=False
)
Required environment variables:

.env

Copy
DEEPSEEK_API_KEY=
​
Ollama
Many users asked for local models. Here they are.

Download Ollama from here
Run ollama pull model_name. Pick a model which supports tool-calling from here
Run ollama start

Copy
from langchain_ollama import ChatOllama
from browser_use import Agent
from pydantic import SecretStr


# Initialize the model
llm=ChatOllama(model="qwen2.5", num_ctx=32000)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)
Required environment variables: None!

​
Coming soon
(We are working on it)

Groq
Github
Fine-tuned models
---

Agent Settings
Learn how to configure the agent

​
Overview
The Agent class is the core component of Browser Use that handles browser automation. Here are the main configuration options you can use when initializing an agent.

​
Basic Settings

Copy
from browser_use import Agent
from langchain_openai import ChatOpenAI

agent = Agent(
    task="Search for latest news about AI",
    llm=ChatOpenAI(model="gpt-4o"),
)
​
Required Parameters
task: The instruction for the agent to execute
llm: A LangChain chat model instance. See LangChain Models for supported models.
​
Agent Behavior
Control how the agent operates:


Copy
agent = Agent(
    task="your task",
    llm=llm,
    controller=custom_controller,  # For custom tool calling
    use_vision=True,              # Enable vision capabilities
    save_conversation_path="logs/conversation"  # Save chat logs
)
​
Behavior Parameters
controller: Registry of functions the agent can call. Defaults to base Controller. See Custom Functions for details.
use_vision: Enable/disable vision capabilities. Defaults to True.
When enabled, the model processes visual information from web pages
Disable to reduce costs or use models without vision support
For GPT-4o, image processing costs approximately 800-1000 tokens (~$0.002 USD) per image (but this depends on the defined screen size)
save_conversation_path: Path to save the complete conversation history. Useful for debugging.
system_prompt_class: Custom system prompt class. See System Prompt for customization options.
Vision capabilities are recommended for better web interaction understanding, but can be disabled to reduce costs or when using models without vision support.

​
(Reuse) Browser Configuration
You can configure how the agent interacts with the browser. To see more Browser options refer to the Browser Settings documentation.

​
Reuse Existing Browser
browser: A Browser Use Browser instance. When provided, the agent will reuse this browser instance and automatically create new contexts for each run().


Copy
from browser_use import Agent, Browser
from browser_use.browser.context import BrowserContext

# Reuse existing browser
browser = Browser()
agent = Agent(
    task=task1,
    llm=llm,
    browser=browser  # Browser instance will be reused
)

await agent.run()

# Manually close the browser
await browser.close()
Remember: in this scenario the Browser will not be closed automatically.

​
Reuse Existing Browser Context
browser_context: A Playwright browser context. Useful for maintaining persistent sessions. See Persistent Browser for more details.


Copy
from browser_use import Agent, Browser
from playwright.async_api import BrowserContext

# Use specific browser context (preferred method)
async with await browser.new_context() as context:
    agent = Agent(
        task=task2,
        llm=llm,
        browser_context=context  # Use persistent context
    )

    # Run the agent
    await agent.run()

    # Pass the context to the next agent
    next_agent = Agent(
        task=task2,
        llm=llm,
        browser_context=context
    )

    ...

await browser.close()
For more information about how browser context works, refer to the Playwright documentation.

You can reuse the same context for multiple agents. If you do nothing, the browser will be automatically created and closed on run() completion.

​
Running the Agent
The agent is executed using the async run() method:

max_steps (default: 100)
Maximum number of steps the agent can take during execution. This prevents infinite loops and helps control execution time.
​
Agent History
The method returns an AgentHistoryList object containing the complete execution history. This history is invaluable for debugging, analysis, and creating reproducible scripts.


Copy
# Example of accessing history
history = await agent.run()

# Access (some) useful information
history.urls()              # List of visited URLs
history.screenshots()       # List of screenshot paths
history.action_names()      # Names of executed actions
history.extracted_content() # Content extracted during execution
history.errors()           # Any errors that occurred
history.model_actions()     # All actions with their parameters
The AgentHistoryList provides many helper methods to analyze the execution:

final_result(): Get the final extracted content
is_done(): Check if the agent completed successfully
has_errors(): Check if any errors occurred
model_thoughts(): Get the agent’s reasoning process
action_results(): Get results of all actions
For a complete list of helper methods and detailed history analysis capabilities, refer to the AgentHistoryList source code.

​
Run initial actions without LLM
With this example you can run initial actions without the LLM. Specify the action as a dictionary where the key is the action name and the value is the action parameters. You can find all our actions in the Controller source code.


Copy

initial_actions = [
  {'open_tab': {'url': 'https://www.google.com'}},
  {'open_tab': {'url': 'https://en.wikipedia.org/wiki/Randomness'}},
  {'scroll_down': {'amount': 1000}},
]
agent = Agent(
  task='What theories are displayed on the page?',
  initial_actions=initial_actions,
  llm=llm,
)
​
Run with message context
You can configure the agent and provide a separate message to help the LLM understand the task better.


Copy
from langchain_openai import ChatOpenAI

agent = Agent(
    task="your task",
    message_context="Additional information about the task",
    llm = ChatOpenAI(model='gpt-4o')
)
​
Run with planner model
You can configure the agent to use a separate planner model for high-level task planning:


Copy
from langchain_openai import ChatOpenAI

# Initialize models
llm = ChatOpenAI(model='gpt-4o')
planner_llm = ChatOpenAI(model='o3-mini')

agent = Agent(
    task="your task",
    llm=llm,
    planner_llm=planner_llm,           # Separate model for planning
    use_vision_for_planner=False,      # Disable vision for planner
    planner_interval=4                 # Plan every 4 steps
)
​
Planner Parameters
planner_llm: A LangChain chat model instance used for high-level task planning. Can be a smaller/cheaper model than the main LLM.
use_vision_for_planner: Enable/disable vision capabilities for the planner model. Defaults to True.
planner_interval: Number of steps between planning phases. Defaults to 1.
Using a separate planner model can help:

Reduce costs by using a smaller model for high-level planning
Improve task decomposition and strategic thinking
Better handle complex, multi-step tasks
The planner model is optional. If not specified, the agent will not use the planner model.

​
Optional Parameters
message_context: Additional information about the task to help the LLM understand the task better.
initial_actions: List of initial actions to run before the main task.
max_actions_per_step: Maximum number of actions to run in a step. Defaults to 10.
max_failures: Maximum number of failures before giving up. Defaults to 3.
retry_delay: Time to wait between retries in seconds when rate limited. Defaults to 10.
generate_gif: Enable/disable GIF generation. Defaults to False. Set to True or a string path to save the GIF.

---

Customize
Browser Settings
Configure browser behavior and context settings

Browser Use allows you to customize the browser’s behavior through two main configuration classes: BrowserConfig and BrowserContextConfig. These settings control everything from headless mode to proxy settings and page load behavior.

We are currently working on improving how browser contexts are managed. The system will soon transition to a “1 agent, 1 browser, 1 context” model for better stability and developer experience.

​
Browser Configuration
The BrowserConfig class controls the core browser behavior and connection settings.


Copy
from browser_use import BrowserConfig

# Basic configuration
config = BrowserConfig(
    headless=False,
    disable_security=True
)

browser = Browser(config=config)

agent = Agent(
    browser=browser,
    # ...
)
​
Core Settings
headless (default: False) Runs the browser without a visible UI. Note that some websites may detect headless mode.

disable_security (default: True) Disables browser security features. While this can fix certain functionality issues (like cross-site iFrames), it should be used cautiously, especially when visiting untrusted websites.

​
Additional Settings
extra_browser_args (default: []) Additional arguments are passed to the browser at launch. See the full list of available arguments.

proxy (default: None) Standard Playwright proxy settings for using external proxy services.

new_context_config (default: BrowserContextConfig()) Default settings for new browser contexts. See Context Configuration below.

For web scraping tasks on sites that restrict automated access, we recommend using external browser or proxy providers for better reliability.

​
Alternative Initialization
These settings allow you to connect to external browser providers or use a local Chrome instance.

​
External Browser Provider (wss)
Connect to cloud-based browser services for enhanced reliability and proxy capabilities.


Copy
config = BrowserConfig(
    wss_url="wss://your-browser-provider.com/ws"
)
wss_url (default: None) WebSocket URL for connecting to external browser providers (e.g., anchorbrowser.io, steel.dev, browserbase.com, browserless.io, TestingBot).
This overrides local browser settings and uses the provider’s configuration. Refer to their documentation for settings.

​
External Browser Provider (cdp)
Connect to cloud or local Chrome instances using Chrome DevTools Protocol (CDP) for use with tools like headless-shell or browserless.


Copy
config = BrowserConfig(
    cdp_url="http://localhost:9222"
)
cdp_url (default: None) URL for connecting to a Chrome instance via CDP. Commonly used for debugging or connecting to locally running Chrome instances.
​
Local Chrome Instance (binary)
Connect to your existing Chrome installation to access saved states and cookies.


Copy
config = BrowserConfig(
    browser_instance_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
browser_instance_path (default: None) Path to connect to an existing Browser installation. Particularly useful for workflows requiring existing login states or browser preferences.
This will overwrite other browser settings.
​
Context Configuration
The BrowserContextConfig class controls settings for individual browser contexts.


Copy
from browser_use.browser.context import BrowserContextConfig

config = BrowserContextConfig(
    cookies_file="path/to/cookies.json",
    wait_for_network_idle_page_load_time=3.0,
    browser_window_size={'width': 1280, 'height': 1100},
    locale='en-US',
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    viewport_expansion=500,
    allowed_domains=['google.com', 'wikipedia.org'],
)

browser = Browser()
context = BrowserContext(browser=browser, config=config)


async def run_search():
  agent = Agent(
    browser_context=context,
    task='Your task',
    llm=llm)
​
Configuration Options
​
Page Load Settings
minimum_wait_page_load_time (default: 0.5) Minimum time to wait before capturing page state for LLM input.

wait_for_network_idle_page_load_time (default: 1.0) Time to wait for network activity to cease. Increase to 3-5s for slower websites. This tracks essential content loading, not dynamic elements like videos.

maximum_wait_page_load_time (default: 5.0) Maximum time to wait for page load before proceeding.

​
Display Settings
browser_window_size (default: {'width': 1280, 'height': 1100}) Browser window dimensions. The default size is optimized for general use cases and interaction with common UI elements like cookie banners.

locale (default: None) Specify user locale, for example en-GB, de-DE, etc. Locale will affect the navigator. Language value, Accept-Language request header value as well as number and date formatting rules. If not provided, defaults to the system default locale.

highlight_elements (default: True) Highlight interactive elements on the screen with colorful bounding boxes.

viewport_expansion (default: 500) Viewport expansion in pixels. With this you can controll how much of the page is included in the context of the LLM. If set to -1, all elements from the entire page will be included (this leads to high token usage). If set to 0, only the elements which are visible in the viewport will be included. Default is 500 pixels, that means that we inlcude a little bit more than the visible viewport inside the context.

​
Restrict URLs
allowed_domains (default: None) List of allowed domains that the agent can access. If None, all domains are allowed. Example: [‘google.com’, ‘wikipedia.org’] - Here the agent will only be able to access google and wikipedia.
​
Debug and Recording
save_recording_path (default: None) Directory path for saving video recordings.

trace_path (default: None) Directory path for saving trace files. Files are automatically named as {trace_path}/{context_id}.zip.

---

Customize
Connect to your Browser
With this you can connect to your real browser, where you are logged in with all your accounts.

​
Overview
You can connect the agent to your real Chrome browser instance, allowing it to access your existing browser profile with all your logged-in accounts and settings. This is particularly useful when you want the agent to interact with services where you’re already authenticated.

First make sure to close all running Chrome instances.

​
Basic Configuration
To connect to your real Chrome browser, you’ll need to specify the path to your Chrome executable when creating the Browser instance:


Copy
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
import asyncio
# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        browser_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS path
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # For Linux, typically: '/usr/bin/google-chrome'
    )
)

# Create the agent with your configured browser
agent = Agent(
    task="Your task here",
    llm=ChatOpenAI(model='gpt-4o'),
    browser=browser,
)

async def main():
    await agent.run()

    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
When using your real browser, the agent will have access to all your logged-in sessions. Make sure to ALWAYS review the task you’re giving to the agent and ensure it aligns with your security requirements!

---

Customize
Output Format
The default is text. But you can define a structured output format to make post-processing easier.

​
Custom output format
With this example you can define what output format the agent should return to you.


Copy
from pydantic import BaseModel
# Define the output format as a Pydantic model
class Post(BaseModel):
  post_title: str
  post_url: str
  num_comments: int
  hours_since_post: int


class Posts(BaseModel):
  posts: List[Post]


controller = Controller(output_model=Posts)


async def main():
  task = 'Go to hackernews show hn and give me the first  5 posts'
  model = ChatOpenAI(model='gpt-4o')
  agent = Agent(task=task, llm=model, controller=controller)

  history = await agent.run()

  result = history.final_result()
  if result:
    parsed: Posts = Posts.model_validate_json(result)

    for post in parsed.posts:
      print('\n--------------------------------')
      print(f'Title:            {post.post_title}')
      print(f'URL:              {post.post_url}')
      print(f'Comments:         {post.num_comments}')
      print(f'Hours since post: {post.hours_since_post}')
  else:
    print('No result')


if __name__ == '__main__':
  asyncio.run(main())

  ---


  Customize
System Prompt
Customize the system prompt to control agent behavior and capabilities

​
Overview
You can customize the system prompt in two ways:

Extend the default system prompt with additional instructions
Override the default system prompt entirely
Custom system prompts allow you to modify the agent’s behavior at a fundamental level. Use this feature carefully as it can significantly impact the agent’s performance and reliability.

​
Extend System Prompt (recommended)
To add additional instructions to the default system prompt:


Copy
from browser_use import Agent
from langchain_openai import ChatOpenAI

# Add your custom instructions
extend_system_message = """
REMEMBER the most important RULE:
ALWAYS open first a new tab and go first to url wikipedia.com no matter the task!!!
"""

# Create agent with extended system prompt
agent = Agent(
    task="Your task here",
    llm=ChatOpenAI(model='gpt-4'),
    extend_system_message=extend_system_message
)
​
Override System Prompt
Not recommended! If you must override the default system prompt, make sure to test the agent yourself.

Anyway, to override the default system prompt:


Copy
# Define your complete custom prompt
override_system_message = """
You are an AI agent that helps users with web browsing tasks.

[Your complete custom instructions here...]
"""

# Create agent with custom system prompt
agent = Agent(
    task="Your task here",
    llm=ChatOpenAI(model='gpt-4'),
    override_system_message=override_system_message
)

---


Customize
Custom Functions
Extend default agent and write custom function calls

​
Basic Function Registration
Functions can be either sync or async. Keep them focused and single-purpose.


Copy
from browser_use import Controller, ActionResult
# Initialize the controller
controller = Controller()

@controller.action('Ask user for information')
def ask_human(question: str) -> str:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)
Basic Controller has all basic functionality you might need to interact with the browser already implemented.


Copy
# ... then pass controller to the agent
agent = Agent(
    task=task,
    llm=llm,
    controller=controller
)
Keep the function name and description short and concise. The Agent use the function solely based on the name and description. The stringified output of the action is passed to the Agent.

​
Browser-Aware Functions
For actions that need browser access, simply add the browser parameter inside the function parameters:


Copy
from browser_use import Browser, Controller, ActionResult

controller = Controller()
@controller.action('Open website')
async def open_website(url: str, browser: Browser):
    page = browser.get_current_page()
    await page.goto(url)
    return ActionResult(extracted_content='Website opened')
​
Structured Parameters with Pydantic
For complex actions, you can define parameter schemas using Pydantic models:


Copy
from pydantic import BaseModel
from typing import Optional
from browser_use import Controller, ActionResult, Browser

controller = Controller()

class JobDetails(BaseModel):
    title: str
    company: str
    job_link: str
    salary: Optional[str] = None

@controller.action(
    'Save job details which you found on page',
    param_model=JobDetails
)
async def save_job(params: JobDetails, browser: Browser):
    print(f"Saving job: {params.title} at {params.company}")

    # Access browser if needed
    page = browser.get_current_page()
    await page.goto(params.job_link)
​
Using Custom Actions with multiple agents
You can use the same controller for multiple agents.


Copy
controller = Controller()

# ... register actions to the controller

agent = Agent(
    task="Go to website X and find the latest news",
    llm=llm,
    controller=controller
)

# Run the agent
await agent.run()

agent2 = Agent(
    task="Go to website Y and find the latest news",
    llm=llm,
    controller=controller
)

await agent2.run()
The controller is stateless and can be used to register multiple actions and multiple agents.

​
Exclude functions
If you want less actions to be used by the agent, you can exclude them from the controller.


Copy
controller = Controller(exclude_actions=['open_tab', 'search_google'])
For more examples like file upload or notifications, visit examples/custom-functions.

---

Development
Local Setup
Set up Browser Use development environment locally

​
Prerequisites
Browser Use requires Python 3.11 or higher. We recommend using uv for Python environment management.

​
Clone the Repository
First, clone the Browser Use repository:


Copy
git clone https://github.com/browser-use/browser-use
cd browser-use
​
Environment Setup
Create a virtual environment:

Copy
uv venv --python 3.11
Install dependencies:

Copy
# Install the package in editable mode with all development dependencies
uv pip install -e ".[dev]"
The -e flag installs the package in “editable” mode, which means your local code changes will be reflected immediately without requiring reinstallation. The [dev] part installs additional dependencies needed for development.

​
Configuration
Set up your environment variables:


Copy
# Copy the example environment file
cp .env.example .env
Or manually create a .env file with your API keys:

.env

Copy
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
You can use any LLM model supported by LangChain. See LangChain Models for available options and their specific API key requirements.

​
Development
After setup, you can:

Run tests with pytest
Build the package with hatch build
Try the examples in the examples/ directory
​
Getting Help
If you run into any issues:

Check our GitHub Issues
Join our Discord community for support

---

Development
Observability
Trace Browser Use’s agent execution steps and browser sessions

​
Overview
Browser Use has a native integration with Laminar - open-source platform for tracing, evals and labeling of AI agents. Read more about Laminar in the Laminar docs.

Laminar excels at tracing browser agents by providing unified visibility into both browser session recordings and agent execution steps.

​
Setup
To setup Laminar, you need to install the lmnr package and set the LMNR_PROJECT_API_KEY environment variable.

To get your project API key, you can either:

Register on Laminar Cloud and get the key from your project settings
Or spin up a local Laminar instance and get the key from the settings page

Copy
pip install 'lmnr[all]'
export LMNR_PROJECT_API_KEY=<your-project-api-key>
​
Usage
Then, you simply initialize the Laminar at the top of your project and both Browser Use and session recordings will be automatically traced.


Copy
from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio

from lmnr import Laminar
# this line auto-instruments Browser Use and any browser you use (local or remote)
Laminar.initialize(project_api_key="...") # you can also pass project api key here

async def main():
    agent = Agent(
        task="open google, search Laminar AI",
        llm=ChatOpenAI(model="gpt-4o-mini"),
    )
    result = await agent.run()
    print(result)

asyncio.run(main())
​
Viewing Traces
You can view traces in the Laminar UI by going to the traces tab in your project. When you select a trace, you can see both the browser session recording and the agent execution steps.

Timeline of the browser session is synced with the agent execution steps, timeline highlights indicate the agent’s current step synced with the browser session. In the trace view, you can also see the agent’s current step, the tool it’s using, and the tool’s input and output. Tools are highlighted in the timeline with a yellow color.

Laminar
​
Laminar
To learn more about tracing and evaluating your browser agents, check out the Laminar docs.

---


