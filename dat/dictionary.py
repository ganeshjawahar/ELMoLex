import json
import os


class Dictionary(object):

  def __init__(self, name, default_value=False, keep_growing=True, singleton=False):
    self.__name = name

    self.instance2index = {}
    self.instances = []
    self.default_value = default_value
    self.offset = 1 if self.default_value else 0
    self.keep_growing = keep_growing
    self.singletons = set() if singleton else None

    # Index 0 is occupied by default, all else following.
    self.default_index = 0 if self.default_value else None

    self.next_index = self.offset
  
  def add(self, instance):
    if instance not in self.instance2index:
      self.instances.append(instance)
      self.instance2index[instance] = self.next_index
      self.next_index += 1

  def add_singleton(self, id):
    if self.singletons is None:
      raise RuntimeError('Dictionary %s does not have singleton.' % self.__name)
    else:
      self.singletons.add(id)

  def get_index(self, instance):
    try:
      return self.instance2index[instance]
    except KeyError:
      if self.keep_growing:
        index = self.next_index
        self.add(instance)
        return index
      else:
        if self.default_value:
          return self.default_index
        else:
          print("DEBUG : pos instance", self.instance2index)
          raise KeyError("instance not found: %s " % instance)

  def get_content(self):
    if self.singletons is None:
      return {'instance2index': self.instance2index, 'instances': self.instances}
    else:
      return {'instance2index': self.instance2index, 'instances': self.instances, 'singletions': list(self.singletons)}

  def save(self, output_directory, name=None):
    saving_name = name if name else self.__name
    try:
      dic_dir = os.path.join(output_directory, saving_name + ".json")
      if os.path.isfile(dic_dir):
        print("Overwriting dictionary {}".format(dic_dir))
      json.dump(self.get_content(), open(dic_dir, 'w'), indent=4)
    except Exception as e:
      raise RuntimeError("Dictionary is not saved: %s" % repr(e))

  def close(self):
    self.keep_growing = False

  def size(self):
    return len(self.instances) + self.offset

  def is_singleton(self, id):
    if self.singletons is None:
      raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
    else:
      return id in self.singletons

  def items(self):
    return self.instance2index.items()

  def get_instance(self, index):
    if self.default_value and index == self.default_index:
      # First index is occupied by the wildcard element.
      return '<_UNK>'
    else:
      try:
        return self.instances[index - self.offset]
      except IndexError:
        raise IndexError('unknown index: %d' % index)
  
  def __from_json(self, data):
    self.instances = data["instances"]
    self.instance2index = data["instance2index"]
    if 'singletions' in data:
      self.singletons = set(data['singletions'])
    else:
      self.singletons = None
  
  def load(self, input_directory, name):
    """
    Load model architecture and weights from the give directory. This allow we use old models even the structure
    changes.
    :param input_directory: Directory to save model and weights
    :return:
    """
    loading_name = name
    self.__from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
    self.next_index = len(self.instances) + self.offset
    self.keep_growing = False






