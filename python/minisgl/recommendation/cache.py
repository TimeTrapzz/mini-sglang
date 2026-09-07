from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field


@dataclass(eq=False)
class Node:
    slot: int = -1
    parent: Node | None = None
    token: int = -1
    users: int = 0
    used: int = 0
    children: dict[int, Node] = field(default_factory=dict)


class PrefixPool:
    """Token radix trie with pinned paths and leaf-LRU eviction, owned by one worker.

    An active request reserves all its decode slots at admission. Each physical slot
    is written once; sibling beams share indices until they write a new token.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.free = deque(range(capacity))
        self.root = Node()
        self.nodes: set[Node] = set()
        self.clock = 0

    def match(self, tokens: list[int], pin: bool = False) -> list[Node]:
        node, path = self.root, []
        for token in tokens:
            node = node.children.get(token)
            if node is None:
                break
            path.append(node)
        if pin:
            self.pin(path)
        return path

    def pin(self, path: list[Node]):
        self.clock += 1
        for node in path:
            node.users += 1
            node.used = self.clock

    def release(self, path: list[Node]):
        for node in path:
            assert node.users > 0
            node.users -= 1

    @property
    def available(self):
        return len(self.free) + sum(n.users == 0 for n in self.nodes)

    def allocate(self, count: int) -> list[int]:
        if count > self.available:
            raise MemoryError("KV pool cannot admit this request")
        candidates = [(n.used, n.slot, n) for n in self.nodes if not n.users and not n.children]
        heapq.heapify(candidates)
        while len(self.free) < count:
            if not candidates:
                raise RuntimeError("Pinned-prefix accounting error")
            _, _, node = heapq.heappop(candidates)
            del node.parent.children[node.token]
            self.nodes.remove(node)
            self.free.append(node.slot)
            parent = node.parent
            if parent is not self.root and not parent.users and not parent.children:
                heapq.heappush(candidates, (parent.used, parent.slot, parent))
        return [self.free.popleft() for _ in range(count)]

    def insert(self, tokens: list[int], slots: list[int]) -> tuple[list[Node], set[int]]:
        node, path, adopted = self.root, [], set()
        for token, slot in zip(tokens, slots, strict=True):
            if token not in node.children:
                child = Node(slot=slot, parent=node, token=token)
                node.children[token] = child
                self.nodes.add(child)
                adopted.add(slot)
            node = node.children[token]
            path.append(node)
        self.pin(path)
        return path, adopted

    def free_owned(self, slots: list[int], adopted: set[int]):
        self.free.extend(slot for slot in slots if slot not in adopted)
