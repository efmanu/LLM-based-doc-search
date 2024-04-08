from llama_index.core.postprocessor import PIINodePostprocessor
from llama_index import ServiceContext, TextNode
from llama_index.schema import NodeWithScore

# Create a service context
service_context = ServiceContext.from_defaults()

# Create a processor
processor = PIINodePostprocessor(service_context=service_context)

# Load document
text = "Hello John Doe. Your phone number is 123-456-7890."
node = TextNode(text=text)

# Process nodes
new_nodes = processor.postprocess_nodes([NodeWithScore(node=node)])

# View redacted text
print(new_nodes[0].node.get_text())