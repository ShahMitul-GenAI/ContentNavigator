import pickle
import shutil
from io import StringIO
import sys
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec


tmp = sys.stdout
my_result = StringIO()
sys.stdout = my_result
print('hello world', file=my_result) # output stored in my_result


print("Hello, how are you?", file=my_result)
print("I am fine. And, you?", file=my_result)

# print(result.getvalue())

with open("./notifications/note1.txt", 'w') as f:
        print(my_result.getvalue(), file=f)
        # my_result.seek(0)
        # shutil.copyfileobj(my_result, f)
f.close()

pickle.dumps("note1.txt")

sys.stdout = tmp

# with open("./notifications/note1.txt", 'r') as fp:
#         lines = fp.splitelines()
#         for each in lines:
#                 print(each)

database = PineConeVectorStore.from_documents(
        documents = documents,
        embedding = embeddings,
        index_name = pnc_indx
    )
