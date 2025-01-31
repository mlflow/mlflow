import locale
import sys

print(sys.getdefaultencoding())  # noqa
print(locale.getpreferredencoding())  # noqa

with open("foo.txt", "w") as f:
    print(f.encoding)  # noqa
    f.write("α")
