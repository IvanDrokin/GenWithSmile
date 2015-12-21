# coding=utf-8


class MetaSmilesFormatError(Exception):
    """
    Бросается при обнаружении ошибки записи строки в формате MetaSMILES
    """
    def __init__(self, expected, found, position, message=None):
        """
        :param expected: tokens.Token -- ожидаемый токен
        :param found: tokens.Token -- обнаруженный токен
        :param position: int -- положение в строке, в котором обнаружена ошибка
        :param message: str -- сообщение об ошибке
        """
        self.expected = expected
        self.found = found
        self.position = position
        self.message = message

    def __str__(self):
        pattern = 'На позиции {self.position} ожидается {self.expected}; найден {self.found}.\n'
        if self.expected == self.found:
            'Токен {self.expected}, позиция {self.position}. '
        if self.message is not None:
            pattern += 'Сообщение: {self.message}'
        return patten.format(self=self)
