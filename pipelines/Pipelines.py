from __future__ import annotations
from operator import length_hint
from typing import *
from ..Serializable import Serializable

Input, Output = TypeVar('Input'), TypeVar('Output')

class Stream(Generic[Output], Serializable):
    # We can't use dataclasses for this and allow the user to set these values in __init__ because we need to allow
    # subclasses be dataclasses and have required fields.
    def __init__(self):
        super().__init__()
        self.iter_counter: int = 0
        self.autosave_interval: Optional[int] = None  # If non-None, will back up state to disk every N data chunks

        self._raw_generator: Optional[Iterator[Output]] = None  # Private variable

    # Useful for, e.g., tqdm progress bars
    def __length_hint__(self):
        return NotImplemented

    def generate(self) -> Iterator[Output]:
        pass

    def __iter__(self):
        return self

    def __next__(self):
        # Lazily create the raw generator
        if not self._raw_generator:
            self._raw_generator = self.generate()

        next_val = next(self._raw_generator)
        self.iter_counter += 1  # Only increment if we didn't just throw StopIteration

        if self.autosave_interval and self.iter_counter % self.autosave_interval == 0:
            self.autosave()

        return next_val


class Target(Generic[Input], Serializable):
    def __init__(self):
        super().__init__()
        self.inputs: Optional[Stream[Input]] = None

    def read(self) -> None:  # Reads and processes input from the Stream
        pass


# A generator that takes another generator as its input
class Filter(Target[Input], Stream[Output]):
    pass


# Pipelines can either be 'open', in which case they're just fancy composite Streams, or 'closed', where their
# outputs are piped into a Target object that just consumes the stream and doesn't yield anything. For closed
# Pipelines you can call Pipeline.run().
class Pipeline(Stream):
    def __init__(self, children: List[Serializable]):
        super(Pipeline, self).__init__()
        self.children = children

        # Make sure each element knows about its predecessor if it needs to
        predecessor = None
        for i, child in enumerate(children):
            child.autosave_interval = None  # We take care of saving for the children, so don't do it redundantly

            # Is this a Filter or a Target?
            if hasattr(child, 'inputs'):
                assert child.inputs or predecessor, f"No stream given to feed component {child} at index {i}."

                # Connect this component to its predecessor
                child.inputs = predecessor
            else:
                assert i == 0, \
                    f"Found Stream {child} at index {i}, but Streams are only allowed at the beginning " \
                    f"of a Pipeline."

            predecessor = child

    def __length_hint__(self):
        return length_hint(self.children[-1])

    def generate(self) -> Iterator[Any]:
        last_component = self.children[-1]

        assert isinstance(last_component, Stream), \
            f"Last component {last_component} must be a Stream in order to iterate over a Pipeline."

        yield from last_component

    def run(self) -> None:
        target = self.children[-1]

        assert isinstance(target, Target), \
            "The last pipeline component must be a Target to use Pipeline.run()."

        target.read()
