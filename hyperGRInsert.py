
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
single_doc = ' '.join(documents)
# an example prompt
prompt = "Give the list of all Knowledge Hypergraph triplets for the following text" +"\n"+single_doc

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama3.1:8b",
  # prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
  prompt=prompt
)

print(output['response'])