"""Document-at-a-time packed token stream with an explicit, restorable cursor."""
import copy

from .corpus import require
from .tracking import digest

VERSION = "aim-packed-token-cursor-v1"


class TokenStream:
    def __init__(self, corpus, tokenizer, split="train", repeat=True, cursor=None):
        require(type(repeat) is bool, "stream repeat must be Boolean")
        self.corpus, self.tokenizer, self.split, self.repeat = corpus, tokenizer, split, repeat
        self.rows = corpus.records(split)
        require(bool(self.rows), "requested stream split is empty")
        self.document_index, self.token_offset, self.epoch, self.carry = 0, 0, 0, None
        self.cached_index, self.cached_tokens = None, None
        self.identity = {"schema": VERSION, "corpus_hash": corpus.fingerprint,
                         "tokenizer_hash": digest(tokenizer.specification()), "split": split, "repeat": repeat}
        if cursor is not None:
            require(isinstance(cursor, dict) and set(cursor) == set(self.cursor()), "invalid token cursor fields")
            require(all(cursor[k] == v for k, v in self.identity.items()), "token cursor corpus/tokenizer/split mismatch")
            for key in ("document_index", "token_offset", "epoch"):
                require(type(cursor[key]) is int and 0 <= cursor[key] <= 10**12, "invalid token cursor coordinate")
            self.document_index, self.token_offset, self.epoch = (cursor[k] for k in ("document_index", "token_offset", "epoch"))
            require(self.document_index < len(self.rows) or (not repeat and self.document_index == len(self.rows)), "token cursor document out of bounds")
            require(repeat or self.epoch == 0, "finite stream cannot have a repeated epoch")
            if self.document_index < len(self.rows):
                require(self.token_offset < len(self._tokens(self.document_index)), "token offset out of bounds")
            else:
                require(self.token_offset == 0, "exhausted finite stream has nonzero offset")
            self.carry = copy.deepcopy(cursor["carry"])
            require(self.carry == self._expected_carry(), "lookahead carry does not match token cursor")

    def _tokens(self, index):
        if index != self.cached_index:
            text = self.corpus.text(self.rows[index])
            self.cached_tokens = [self.tokenizer.bos_id]+self.tokenizer.encode(text)+[self.tokenizer.eos_id]
            self.cached_index = index
        return self.cached_tokens

    def _expected_carry(self):
        index, offset, epoch = self.document_index, self.token_offset, self.epoch
        if index == offset == epoch == 0:
            return None
        if offset:
            offset -= 1
        else:
            index -= 1
            if index < 0:
                index = len(self.rows)-1
                epoch -= 1
            offset = len(self._tokens(index))-1
        return [self._tokens(index)[offset], self.rows[index]["id"], offset, epoch]

    def _next(self):
        if self.document_index == len(self.rows):
            raise StopIteration
        tokens = self._tokens(self.document_index)
        item = [tokens[self.token_offset], self.rows[self.document_index]["id"], self.token_offset, self.epoch]
        self.token_offset += 1
        if self.token_offset == len(tokens):
            self.token_offset = 0
            self.document_index += 1
            if self.document_index == len(self.rows) and self.repeat:
                self.document_index = 0
                self.epoch += 1
        return item

    def cursor(self):
        return {**self.identity, "document_index": self.document_index, "token_offset": self.token_offset,
                "epoch": self.epoch, "carry": copy.deepcopy(self.carry)}

    def batch(self, batch_size, context):
        require(type(batch_size) is int and 1 <= batch_size <= 16 and type(context) is int and 1 <= context <= 512,
                "stream batch/context outside miniature bounds")
        xs, ys, masks, provenance = [], [], [], []
        for _ in range(batch_size):
            items = []
            if self.carry is not None:
                items.append(self.carry)
            try:
                while len(items) < context+1:
                    items.append(self._next())
            except StopIteration:
                pass
            if len(items) < 2:
                break
            self.carry = items[-1]
            tokens = [item[0] for item in items]
            padding = context+1-len(tokens)
            tokens += [self.tokenizer.pad_id]*padding
            xs.append(tokens[:-1]); ys.append(tokens[1:])
            # BOS is a boundary marker, never a next-document prediction target.
            # Causal attention is allowed across packed document boundaries.
            masks.append([int(t not in {self.tokenizer.bos_id, self.tokenizer.pad_id}) for t in tokens[1:]])
            provenance.append(items)
        if not xs:
            raise StopIteration
        return {"inputs": xs, "targets": ys, "mask": masks, "provenance": provenance}
