from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelInfo(_message.Message):
    __slots__ = ["id", "config"]
    ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    id: JobID
    config: str
    def __init__(self, id: _Optional[_Union[JobID, _Mapping]] = ..., config: _Optional[str] = ...) -> None: ...

class Target(_message.Message):
    __slots__ = ["name", "id"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    id: int
    def __init__(self, name: _Optional[str] = ..., id: _Optional[int] = ...) -> None: ...

class Training(_message.Message):
    __slots__ = ["id", "status"]
    ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    id: int
    status: str
    def __init__(self, id: _Optional[int] = ..., status: _Optional[str] = ...) -> None: ...

class JobID(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class Timestamp(_message.Message):
    __slots__ = ["timestamp", "model_info"]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
    timestamp: str
    model_info: str
    def __init__(self, timestamp: _Optional[str] = ..., model_info: _Optional[str] = ...) -> None: ...

class Progress(_message.Message):
    __slots__ = ["id", "data"]
    class DataEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    id: int
    data: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[int] = ..., data: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Results(_message.Message):
    __slots__ = ["target", "metrics"]
    class MetricsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Predictions
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Predictions, _Mapping]] = ...) -> None: ...
    TARGET_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    target: str
    metrics: _containers.MessageMap[str, Predictions]
    def __init__(self, target: _Optional[str] = ..., metrics: _Optional[_Mapping[str, Predictions]] = ...) -> None: ...

class AllResults(_message.Message):
    __slots__ = ["results"]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[Results]
    def __init__(self, results: _Optional[_Iterable[_Union[Results, _Mapping]]] = ...) -> None: ...

class Predictions(_message.Message):
    __slots__ = ["predictions", "evaluation"]
    class EvaluationEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_FIELD_NUMBER: _ClassVar[int]
    predictions: _containers.RepeatedScalarFieldContainer[float]
    evaluation: _containers.ScalarMap[str, float]
    def __init__(self, predictions: _Optional[_Iterable[float]] = ..., evaluation: _Optional[_Mapping[str, float]] = ...) -> None: ...

class Inference(_message.Message):
    __slots__ = ["predictions"]
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    predictions: bytes
    def __init__(self, predictions: _Optional[bytes] = ...) -> None: ...
