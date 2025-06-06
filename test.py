from chunking import chunker

chunk =chunker()

content = chunk.getFile("./test.txt")

chunk.print_log(chunk.default_split(content, 128, 20))

def test(**kwargs):
    print(kwargs.get("name"))

dir = {}
print(dir.get("111"))