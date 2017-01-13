class className:
    def createName(self, name):
        self.name = name
    def displayName(self):
        return self.name
    def saying(self):
        print ("hello %s" % self.name)

first=className()
second=className()

first.createName('bob')
second.createName('tony')

print(first.displayName())
first.saying()