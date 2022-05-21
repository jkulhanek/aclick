API
===

.. module:: aclick

This part of the documentation lists the full API reference of all public
classes and functions. For more details refer to `Click documentation <https://click.palletsprojects.com/>`_

Decorators
----------

.. autofunction:: command

.. autofunction:: group

.. autofunction:: configuration_option

.. autofunction:: aclick.utils.copy_signature

.. autofunction:: aclick.utils.default_from_str


Commands
--------

.. autoclass:: Command
   :members:

.. autoclass:: Group
   :members:


Types
-----

.. autodata:: aclick.types.List

.. autodata:: aclick.types.ClassUnion

.. autodata:: aclick.types.Tuple

.. autodata:: aclick.types.ParameterGroup

.. autodata:: aclick.types.ClassHierarchicalOption

.. autodata:: aclick.types.UnionTypeHierarchicalOption

.. autodata:: aclick.types.OptionalTypeHierarchicalOption


Utilities
---------

.. autofunction:: aclick.utils.parse_class_structure

.. autofunction:: aclick.utils.as_dict

.. autofunction:: aclick.utils.from_dict


Other classes
-------------

.. autoclass:: aclick.ParameterRenamer

.. autoclass:: aclick.RegexParameterRenamer

.. autoclass:: aclick.FlattenParameterRenamer
