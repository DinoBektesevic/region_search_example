from sqlalchemy.types import (
    TypeDecorator,
    Double,
    Integer,
)

# SQLAlch 2.0 seems to come with some bright and shiny new toys but also
# NumPy and Python types are regarded as distinct types. This is ok, because
# I guess we can re-purpose the decorators as AstroPy quantity and unit
# handlers later too
# https://github.com/sqlalchemy/sqlalchemy/issues/5167
# https://github.com/sqlalchemy/sqlalchemy/issues/3586
# https://github.com/sqlalchemy/sqlalchemy/issues/5552
# https://github.com/sqlalchemy/sqlalchemy/discussions/5948


class NpFloat(TypeDecorator):
    """
    A more generic Float type that supports inserting of NumPy `np.float*`
    types by coercing them to a Python builtin `float` type.

    The coercion occurs at statement execution time.
    """
    impl = Double
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return float(value)


class NpInt(TypeDecorator):
    """
    A more generic Integer type that supports inserting of NumPy `np.int*`
    types by coercing them to a Python builtin `int` type.

    The coercion occurs at statement execution time.
    """
    # coerces given data to float() before storing
    # to avoid explosions due to np.int types
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return int(value)

