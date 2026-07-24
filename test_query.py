from query import query_pdf

result = query_pdf(
    "What's the most important task you should do first thing in the morning?",
    filename="Brian_Tracy_Eat_That_Frog.pdf"
)
print(result["answer"])