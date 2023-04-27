import os
import gradio as gr
from src.util import agent_logs


class MyLangchainUI:
    """a simple and awesome ui to display agent actions and thought processes"""

    gradio_app = None

    def __init__(self, func, ui_type="agent_executor_mrkl"):
        # clear old logs
        agent_logs.clear_log()
        # initialize app layouts
        if ui_type == "agent_executor_mrkl":
            self.gradio_app = self._init__agent_executor_mrkl(
                self._clear_log_before_func(func)
            )

    @staticmethod
    def _clear_log_before_func(func):
        def inner1(prompt):
            # clear old logs
            agent_logs.clear_log()
            func(prompt)

        return inner1

    def _init__agent_executor_mrkl(self, func):
        # resource:
        # - https://gradio.app/theming-guide/#discovering-themes
        # - https://gradio.app/quickstart/#more-complexity
        # - https://gradio.app/reactive-interfaces/
        # - https://gradio.app/blocks-and-event-listeners/
        # - https://gradio.app/docs/#highlightedtext

        with gr.Blocks(theme="sudeepshouche/minimalist") as demo:
            gr.Markdown("# A conversation with a generative agent")
            thought_out = None
            # with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    text_input = gr.Textbox(
                        label="Query",
                        info="Ask a question or give an instruction",
                        placeholder="Enter here",
                        lines=3,
                        # value="",
                        examples=[
                            ["What a beautiful morning for a walk!"],
                            ["It was the best of times, it was the worst of times."],
                        ],
                    )
                    # thought_out = gr.Textbox(lines=10, label="Thoughts",
                    #     scroll_to_output=True,)
                    # thought_out = gr.HighlightedText(
                    #     label="Thought Process",
                    #     combine_adjacent=False,
                    #     show_legend=False,
                    #     scroll_to_output=True,
                    # ).style(color_map={"+": "red", "-": "green"})
                    text_output = gr.Textbox(lines=5, label="Final Answer")
                    text_button = gr.Button("Run")

                with gr.Column(scale=1, min_width=600):
                    thought_out = gr.HTML(
                        label="Thought Process", scroll_to_output=True
                    )
                    text_input.change(
                        self.get_thought_process_log,
                        inputs=[],
                        outputs=thought_out,
                        queue=True,
                        every=1,
                    )
            text_button.click(func, inputs=text_input, outputs=text_output)
        return demo

    def get_thought_process_log(self):
        langchain_log = agent_logs.read_log()
        process_html = langchain_log
        # clean up new lines
        process_html = (
            process_html.replace(" \n", "\n")
            .replace("\n\n\n", "\n")
            .replace("\n\n", "\n")
            .replace(": \n", ": ")
            .replace(":\n", ": ")
        )
        # convert new lines to html
        process_html = process_html.replace("\n", "<br>")
        # add colors to different content
        # https://htmlcolors.com/color-names
        # color Tools Available Black
        process_html = process_html.replace(
            "Tools available:", """<p style="color:Black;">Tools available:"""
        )
        # color Question Black
        process_html = process_html.replace(
            "Question:", """</p><p style="color:Black;">Question:"""
        )
        # color Thought Medium Forest Green
        process_html = process_html.replace(
            "Thought:", """</p><p style="color:#348017;">Thought:"""
        )
        # color Action Bee Yellow
        process_html = process_html.replace(
            "Action:", """</p><p style="color:#E9AB17;">Action:"""
        )
        # color Action Bee Yellow
        process_html = process_html.replace(
            "Action Input:", """</p><p style="color:#E9AB17;">Action Input:"""
        )
        # color Observation Denim Dark Blue
        process_html = process_html.replace(
            "Observation:", """</p><p style="color:#151B8D;">Observation:"""
        )
        # color Observation Black
        process_html = process_html.replace(
            "Final Answer:", """</p><p style="color:Black;">Final Answer:"""
        )
        # add closing p
        process_html = f"""{process_html}</p>"""
        return process_html

    def launch(self, server_name="0.0.0.0", server_port=7860):
        self.gradio_app.queue().launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":

    def test_func(prompt):
        answer = f"Question: {prompt}\nThis is a test output."
        import time

        time.sleep(5)
        return answer

    # test this class
    ui_test = MyLangchainUI(test_func)
    ui_test.launch(server_name="0.0.0.0", server_port=7860)
