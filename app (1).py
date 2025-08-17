import os
import asyncio
from dotenv import load_dotenv
import gradio as gr
from sidekick_tools import create_calendar_event, list_upcoming_events
from sidekick import Sidekick

load_dotenv(override=True)

# ------------------------------
# Async setup for Sidekick
# ------------------------------
async def setup_async():
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick

def setup():
    return asyncio.run(setup_async())

# ------------------------------
# Process message
# ------------------------------
async def process_message_async(sidekick, message, success_criteria, history):
    results = await sidekick.run_superstep(message, success_criteria, history or [])
    
    # Extract user and assistant messages from the results
    chat_history = history or []
    if len(results) >= 3:
        user_msg = results[-3]
        assistant_msg = results[-2]
        if isinstance(user_msg, dict) and "content" in user_msg:
            chat_history.append({"role": "user", "content": user_msg["content"]})
        else:
            chat_history.append({"role": "user", "content": str(user_msg)})
        if isinstance(assistant_msg, dict) and "content" in assistant_msg:
            chat_history.append({"role": "assistant", "content": assistant_msg["content"]})
        else:
            chat_history.append({"role": "assistant", "content": str(assistant_msg)})
    return chat_history, sidekick

def process_message(sidekick, message, success_criteria, history):
    return asyncio.run(process_message_async(sidekick, message, success_criteria, history))

# ------------------------------
# Reset sidekick
# ------------------------------
async def reset_async():
    new_sidekick = Sidekick()
    await new_sidekick.setup()
    return "", "", [], new_sidekick

def reset():
    return asyncio.run(reset_async())

# ------------------------------
# Cleanup resources
# ------------------------------
def free_resources(sidekick):
    print("Cleaning up")
    try:
        if sidekick and hasattr(sidekick, "cleanup"):
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")

# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks(title="Sidekick", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    gr.Markdown("## 🤖 Sidekick Personal Co-Worker")
    
    # State does NOT accept visible or delete_callback arguments
    sidekick = gr.State(value=None)
    
    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=300, type="messages")
    
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
        with gr.Row():
            success_criteria = gr.Textbox(show_label=False, placeholder="What are your success criteria?")
    
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")
    
    # Calendar Accordion
    with gr.Accordion("📆 Calendar", open=False):
        cal_summary     = gr.Textbox(label="Event Title")
        cal_start       = gr.Textbox(label="Start (RFC3339)", placeholder="2025-05-20T15:00:00+05:30")
        cal_end         = gr.Textbox(label="End (RFC3339)", placeholder="2025-05-20T16:00:00+05:30")
        cal_description = gr.Textbox(label="Description (optional)")
        add_event_btn   = gr.Button("Add Event")
        list_events_btn = gr.Button("List Upcoming Events")
        cal_output      = gr.Textbox(label="Calendar Output", interactive=False)
    
    # Bind main functions
    ui.load(setup, [], sidekick)
    
    message.submit(fn=process_message, inputs=[sidekick, message, success_criteria, chatbot], outputs=[chatbot, sidekick], queue=True)
    success_criteria.submit(fn=process_message, inputs=[sidekick, message, success_criteria, chatbot], outputs=[chatbot, sidekick], queue=True)
    go_button.click(fn=process_message, inputs=[sidekick, message, success_criteria, chatbot], outputs=[chatbot, sidekick], queue=True)
    
    reset_button.click(fn=reset, inputs=[], outputs=[message, success_criteria, chatbot, sidekick], queue=False)
    
    # Calendar buttons
    add_event_btn.click(
        fn=lambda summary, start, end, desc: create_calendar_event(summary, start, end, desc, os.getenv("GOOGLE_CALENDAR_ID", "primary")),
        inputs=[cal_summary, cal_start, cal_end, cal_description],
        outputs=cal_output,
        queue=False,
    )
    
    list_events_btn.click(
        fn=lambda: list_upcoming_events(os.getenv("GOOGLE_CALENDAR_ID", "primary")),
        outputs=cal_output,
        queue=False,
    )

ui.launch()
