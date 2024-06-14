import pandas as pd
from llm import groq_chat_completion, setup_groq_client

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation.embedding_distance.base import EmbeddingDistanceEvalChain
from langchain.evaluation import load_evaluator

if __name__ == '__main__':
    df = pd.read_csv('car_insurance_queries.csv')
    groq_client = setup_groq_client('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')

    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    hf_evaluator = load_evaluator("embedding_distance", embeddings=hf_embeddings)

    accuracy_criteria = {
        "accuracy": """
    Score 1: The answer is completely unrelated to the reference.
    Score 3: The answer has minor relevance but does not align with the reference.
    Score 5: The answer has moderate relevance but contains inaccuracies.
    Score 7: The answer aligns with the reference but has minor errors or omissions.
    Score 10: The answer is completely accurate and aligns perfectly with the reference."""
    }

    # global groq_client
    labeled_evaluator = load_evaluator(
        "labeled_score_string",
        criteria=accuracy_criteria,
        llm=groq_client,
    )

    response_list = []
    embedding_scores1_list = []
    embedding_scores21_list, embedding_scores22_list = [], []
    for row in df.iterrows():
        print(row[1]['Queries'])
        query = [{"role": "user",
                  "content": row[1]['Queries']}]
        response = groq_chat_completion([], query, doc_type='pdf',
                                        uploaded_file='policy-booklet-0923.pdf')
        print(response)
        response_list.append(response)
        score1 = hf_evaluator.evaluate_strings(
            prediction=response, reference=row[1]['Human Responses']
        )
        score2 = labeled_evaluator.evaluate_strings(
            prediction=response,
            reference=row[1]['Human Responses'],
            input=row[1]['Queries']
        )
        embedding_scores1_list.append(score1['score'])
        embedding_scores21_list.append(score2['score'])
        embedding_scores22_list.append(score2)
        print(score2)
        # break
    df['response'] = response_list  # * len(df)
    df['embedding_scores'] = embedding_scores1_list
    df['reasoning_scores'] = embedding_scores21_list
    df['reasoning_comment'] = embedding_scores22_list
    df.to_csv('car_insurance_queries_with_response.csv')

    
    # chain = EmbeddingDistanceEvalChain()
    
