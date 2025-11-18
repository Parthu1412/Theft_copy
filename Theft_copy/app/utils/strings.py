import logging
import struct as _struct

logger = logging.getLogger()


class StringSerializer(object):
    """
    Serializes unicode to bytes per the configured codec. Defaults to ``utf_8``.

    Note:
        None objects are represented as Kafka Null.

    Args:
        codec (str, optional): encoding scheme. Defaults to utf_8

    See Also:
        `Supported encodings <https://docs.python.org/3/library/codecs.html#standard-encodings>`_

        `StringSerializer Javadoc <https://docs.confluent.io/current/clients/javadocs/org/apache/kafka/common/serialization/StringSerializer.html>`_
    """  # noqa: E501

    def __init__(self, codec="utf_8"):
        self.codec = codec

    def __call__(self, obj, ctx=None):
        """
        Serializes a str(py2:unicode) to bytes.

        Compatibility Note:
            Python 2 str objects must be converted to unicode objects.
            Python 3 all str objects are already unicode objects.

        Args:
            obj (object): object to be serialized

            ctx (SerializationContext): Metadata pertaining to the serialization
                operation

        Raises:
            SerializerError if an error occurs during serialization.

        Returns:
            serialized bytes if obj is not None, otherwise None
        """

        if obj is None:
            return None

        try:
            return obj.encode(self.codec)
        except _struct.error as e:
            logger.error(e)
