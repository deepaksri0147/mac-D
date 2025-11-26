import ollama
import chromadb

documents = [
  "Bears are carnivoran mammals of the family Ursidae.",
  "They are classified as caniforms, or doglike carnivorans.",
  "Although only eight species of bears are extant, they are widespread, appearing in a wide variety of habitats throughout most of the Northern Hemisphere and partially in the Southern Hemisphere.",
  "Bears are found on the continents of North America, South America, and Eurasia.",
  "Common characteristics of modern bears include large bodies with stocky legs, long snouts, small rounded ears, shaggy hair, plantigrade paws with five nonretractile claws, and short tails.",
  "With the exception of courting individuals and mothers with their young, bears are typically solitary animals.",
  "They may be diurnal or nocturnal and have an excellent sense of smell.",
  "Despite their heavy build and awkward gait, they are adept runners, climbers, and swimmers.",
  "Bears use shelters, such as caves and logs, as their dens; most species occupy their dens during the winter for a long period of hibernation, up to 100 days.",
]
Kg_triplets = [["Bears", "belong to", "Family Ursidae"],
["Bears", "classified as", "Caniforms"],
["Bears", "number of species", "Eight"],
["Bears", "habitat", "Northern Hemisphere"],
["Bears", "habitat", "Southern Hemisphere"],
["Bears", "found in", "North America"],
["Bears", "found in", "South America"],
["Bears", "found in", "Eurasia"],
["Modern bears", "characteristic", "Large bodies"],
["Modern bears", "characteristic", "Stocky legs"],
["Modern bears", "characteristic", "Long snouts"],
["Modern bears", "characteristic", "Small rounded ears"],
["Modern bears", "characteristic", "Shaggy hair"],
["Modern bears", "characteristic", "Plantigrade paws with five nonretractile claws"],
["Modern bears", "characteristic", "Short tails"],
["Bears", "social behavior", "Solitary except courting and mothers with young"],
["Bears", "activity pattern", "Diurnal"],
["Bears", "activity pattern", "Nocturnal"],
["Bears", "sense", "Excellent smell"],
["Bears", "capability", "Adept runners"],
["Bears", "capability", "Adept climbers"],
["Bears", "capability", "Adept swimmers"],
["Bears", "use", "Shelters such as caves and logs"],
["Bears", "denning behavior", "Winter hibernation for up to 100 days"],]
# Convert the triplets to text
def triplet_to_text(triplet):
  txt = str(triplet[0]) +" "+str(triplet[1]) +" "+str(triplet[2])
  # print(txt)
  return txt
triplet_texts = [triplet_to_text(triplet) for triplet in Kg_triplets]
# Create database
client = chromadb.PersistentClient(path="database_tmp1")
collection = client.create_collection(name="bear_hkg")
metadata = {"hnsw:space":"cosine"}
# store each document in a vector embedding database

for d in range(0,len(Kg_triplets)):
  triplet_txt = triplet_to_text(Kg_triplets[d])
  response = ollama.embeddings(model="mxbai-embed-large", prompt=triplet_txt)
  embedding = response["embedding"]
  collection.add(
    ids=[str(d)],
    embeddings=[embedding],
    documents=[triplet_txt]
  )

# an example prompt
prompt = "How does the bear's body looks?"

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="mxbai-embed-large"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=3
)

print(collection.get(include=['embeddings','documents','metadatas']))
print("result = ",results)

# data = results['documents'][0][0]
data = ""
supported_docs = results['documents']
if len(supported_docs)==1:
  data = results['documents'][0][0]
else:
  for i in range(0, len(supported_docs)):
    data = data+" "+str(supported_docs[i])
    data = data.strip()
# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama3.1:8b",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print("========================")
print(output['response'])
















# import ollama
# import chromadb

# documents = [
#   "Bears are carnivoran mammals of the family Ursidae.",
#   "They are classified as caniforms, or doglike carnivorans.",
#   "Although only eight species of bears are extant, they are widespread, appearing in a wide variety of habitats throughout most of the Northern Hemisphere and partially in the Southern Hemisphere.",
#   "Bears are found on the continents of North America, South America, and Eurasia.",
#   "Common characteristics of modern bears include large bodies with stocky legs, long snouts, small rounded ears, shaggy hair, plantigrade paws with five nonretractile claws, and short tails.",
#   "With the exception of courting individuals and mothers with their young, bears are typically solitary animals.",
#   "They may be diurnal or nocturnal and have an excellent sense of smell.",
#   "Despite their heavy build and awkward gait, they are adept runners, climbers, and swimmers.",
#   "Bears use shelters, such as caves and logs, as their dens; most species occupy their dens during the winter for a long period of hibernation, up to 100 days.",
# ]
# single_doc = ' '.join(documents)
# # an example prompt
# prompt = "Give the list of all Knowledge Hypergraph triplets for the following text" +"\n"+single_doc

# # generate a response combining the prompt and data we retrieved in step 2
# output = ollama.generate(
#   model="llama3.1:8b",
#   # prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
#   prompt=prompt
# )

# print(output['response'])