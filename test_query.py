from query import query_pdf

result = query_pdf(
    "what is the credit value for CST424",
    filename="PROGRAMMING PARADIGMS (2).pdf"
)
print(result["answer"])