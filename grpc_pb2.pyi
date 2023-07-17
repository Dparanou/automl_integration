from google.protobuf import any_pb2 as _any_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrainingInfo(_message.Message):
    __slots__ = ["id", "config"]
    ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    id: str
    config: str
    def __init__(self, id: _Optional[str] = ..., config: _Optional[str] = ...) -> None: ...

class Target(_message.Message):
    __slots__ = ["name", "id"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    id: str
    def __init__(self, name: _Optional[str] = ..., id: _Optional[str] = ...) -> None: ...

class Status(_message.Message):
    __slots__ = ["id", "status"]
    ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    id: str
    status: str
    def __init__(self, id: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class JobID(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class Timestamp(_message.Message):
    __slots__ = ["timestamp", "model_name"]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    timestamp: str
    model_name: str
    def __init__(self, timestamp: _Optional[str] = ..., model_name: _Optional[str] = ...) -> None: ...

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
    id: str
    data: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., data: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Predictions(_message.Message):
    __slots__ = ["predictions", "timestamps", "evaluation"]
    class EvaluationEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_FIELD_NUMBER: _ClassVar[int]
    predictions: _containers.RepeatedScalarFieldContainer[float]
    timestamps: _containers.RepeatedScalarFieldContainer[float]
    evaluation: _containers.ScalarMap[str, float]
    def __init__(self, predictions: _Optional[_Iterable[float]] = ..., timestamps: _Optional[_Iterable[float]] = ..., evaluation: _Optional[_Mapping[str, float]] = ...) -> None: ...

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

class Inference(_message.Message):
    __slots__ = ["predictions"]
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    predictions: _any_pb2.Any
    def __init__(self, predictions: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ["model_type", "model_name", "target"]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    model_type: str
    model_name: str
    target: str
    def __init__(self, model_type: _Optional[str] = ..., model_name: _Optional[str] = ..., target: _Optional[str] = ...) -> None: ...
