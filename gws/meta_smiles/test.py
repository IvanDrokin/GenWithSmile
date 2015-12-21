import parser

print(parser.parse('CCCCC'))
print(parser.parse('C1C(O)-{remove:1}C1'))
print(parser.parse('CC{attach:fixed:-=#}={remove:1}CN'))
print(parser.parse('C1C{insert:fixed}(O{attach:fixed:-})-{remove:1}C1'))
print(parser.parse('C={remove:2}1CCCCC={remove:2}1'))
print(parser.parse('C1CCF/{remove:1}C=C/FCC1'))
print(parser.parse('NS(O)(={remove:1}O)CN'))
print(parser.parse('CC{attach:fixed:-=}{insert:fixed}CN'))