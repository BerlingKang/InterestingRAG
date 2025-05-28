from re import sub

class chunker:
    def __init__(self):
        self.number = 1

    def readFile(self, path:str, encoding="UTF-8"):
        output = []
        with open(path, encoding=encoding) as file:
            while True:
                line = file.readline()
                if line == "":
                    break
                line = sub("\n", "", line)
                output.append(line)
        return output

    def getFile(self, path, encoding='UTF-8'):
        with open(path, 'r', encoding=encoding) as file:
            return {"text": file.read(), "title": file.name}

    def export_chunks(self, chunks, title):
        export_chunk = []
        for chunk in chunks:
            export_chunk.append({
                "title": title,
                "chunk": chunk
            })
        return export_chunk

    def print_log(self, chunks:dir):
        for chunk in chunks:
            title = chunk["title"]
            print(title)
            log = chunk["chunk"]
            print("content:%s\nstart:%d\nend:%d\n" % (log["content"], log["start_char"], log["end_char"]))

    def default_split(self, content, chunk_size, over_loop, min_size=0):
        text = content.get("text")
        chunks = []
        start = 0

        while start < len(text):
            end = min(start+chunk_size, len(text))
            if end-start<min_size:
                line = text[start:end] + " " * (min_size - end + start)
            else:
                line = text[start:end]
            chunks.append({
                "content": line,
                "start_char": start,
                "end_char":end-1
            })
            start += chunk_size-over_loop
        return self.export_chunks(chunks, content.get("title"))

    def split_by_character(self, content, character):
        text = content.get("text")
        lines = text.split(character)
        chunks = []
        start = 0
        end = 0

        for line in lines:
            end += len(line) + 1
            chunks.append({
                "content": text[start:end],
                "start_char": start,
                "end_char": end-1
            })
            start = end
        return self.export_chunks(chunks, content.get("title"))