# coding=utf-8


class StarSmilesFormatError(Exception):
    def __init__(self, expected, found, position, message=None):
        self.expected = expected
        self.found = found
        self.position = position
        self.message = message

    def __str__(self):
        default_patten = ('На позиции {self.position} ожидается '
                          '{self.expected}, найдено {self.found}.\n')
        if self.message is None:
            return default_patten.format(self=self)
        return (default_patten +
                'Сообщение: {self.message}').format(self=self)
