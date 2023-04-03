from pathlib import Path
from langchain.llms import HuggingFacePipeline
from transformers import (
    # AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from langchain.embeddings import HuggingFaceEmbeddings
from src.my_langchain_docs import MyLangchainDocsHandler


class MyLangchainAgentHandler:
    """a wrapper to make creating a langchain agent easier"""

    def __init__(self) -> None:
        """init function"""
        self.model_name = None
        self.max_new_tokens = None
        self.model = None
        self.tokenizer = None
        self.embedding = None
        self.hf = None

    def load_llama_llm(self, model_name=None, max_new_tokens=50):
        """quick loader of a local llama model

        Args:
            model_name (str, optional): the model name of llama or the folder name with models folder. Defaults to None.
            max_new_tokens (int, optional): max token size used for model. Defaults to 50.

        Returns:model_asset_tuple: a size three tuple of HuggingFacePipeline, model and tokenizer
        """
        if model_name == None:
            self.model_name = "llama-7b"
        else:
            self.model_name = model_name

        model_path = f"models/{self.model_name}"
        self.max_new_tokens = max_new_tokens

        # TODO: review config.json with model folder:
        # https://huggingface.co/docs/transformers/v4.27.2/en/internal/generation_utils#transformers.TemperatureLogitsWarper

        self.tokenizer = AutoTokenizer.from_pretrained(Path(f"{model_path}/"))
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            Path(model_path),
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        # model = model.cuda()
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
        )
        self.hf = HuggingFacePipeline(pipeline=pipe)
        return (self.hf, self.model, self.tokenizer)

    def load_embedding(self):
        """load default embedding used

        Returns:
            HuggingFaceEmbeddings: hugging face embedding model
        """

        # using the default embedding model from hf
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        return self.embedding


if __name__ == "main":

    # # test embeddings
    # # text = "This is a test document."
    # # query_result = embeddings.embed_query(text)
    # # doc_result = embeddings.embed_documents([text])
    # # example_embed1 = embeddings.embed_query("This is a very long sentence and the only difference is a period at the end")
    # # example_embed2 = embeddings.embed_query("This is a very long sentence and the only difference is a period at the end.")
    # # from scipy.spatial import distance
    # # print(f"{distance.euclidean(example_embed1, example_embed2)}")

    # # simple text gen
    # text = "Jim is a helpful business analyst that gives simple, practical answers to questions. \n Bob: What would be a good company name for a company that makes colorful socks? Give me a list of ideas. \n Jim: "
    # # text_response = hf(text)
    # # print(f"{text_response}\n")
    # # print(f"{'='*10}\n")
    # # text_response = hf(text)
    # # print(f"{text_response}\n")
    # # print(f"{'='*10}\n")

    # # query = "What did the president say about Ketanji Brown Jackson"
    # # docs = db.similarity_search(query)

    # # ask some questions about the documents
    # query = "What did the president say about Ketanji Brown Jackson"
    # doc_response = index.query(query, llm=hf)
    # print(f"Query - {query}\nResponse - \n{doc_response}")

    # query = "What are some diagnostic features of substance use disorders?"
    # doc_response = index.query(query, llm=hf)
    # print(f"Query - {query}\nResponse - \n{doc_response}")

    # query = "What data sources are used by llama?"
    # doc_response = index.query(query, llm=hf)
    # print(f"Query - {query}\nResponse - \n{doc_response}")

    # query = "What method did llama use to tokenize its data?"
    # doc_response = index.query(query, llm=hf)
    # print(f"Query - {query}\nResponse - \n{doc_response}")

    # # index.query_with_sources(query)

    # # from langchain.chains.question_answering import load_qa_chain

    # # chain = load_qa_chain(hf, chain_type="stuff")
    # # chain.run(input_documents=docs, question=query)

    # # text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=100, chunk_overlap=0)
    # # texts = text_splitter.split_text(state_of_the_union)
    pass
