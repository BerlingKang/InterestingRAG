from chunking import chunker

chunk =chunker()

content = chunk.getFile("./test.txt")

chunk.print_log(chunk.default_split(content, 128, 20))