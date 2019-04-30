"""
Maps German or English relationships to one of the defined relation types

- SemEval relation types:
Cause-Effect (CE): those <e1>cancers</e1> were caused by radiation <e2>exposures</e2>
Instrument-Agency (IA): <e1>phone</e1> <e2>operator</e2>
Product-Producer (PP): a <e1>factory</e2> manufactures <e2>suits</e2>
Content-Container (CC): a <e1>bottle</e1> full of <e2>honey</e2> was weighed
Entity-Origin (EO): <e1>letters</e1> from foreign <e2>countries</e2>
Entity-Destination (ED): the <e1>boy</e1> went to <e2>bed</e2>
Component-Whole (CW): my <e1>apartment</e1> has a large <e2>kitchen</e2>
Member-Collection (MC): there are many <e1>trees</e1> in the <e2>forest</e2>
Message-Topic (MT): the <e1>lecture</e1> was about <e2>semantics</e2>
Other
"""


class RelationTypes:

    @staticmethod
    def label_to_number(category_name):
        if category_name == 'Cause-Effect':
            return 0
        elif category_name == 'Instrument-Agency':
            return 1
        elif category_name == 'Product-Producer':
            return 2
        elif category_name == 'Content-Container':
            return 3
        elif category_name == 'Entity-Origin':
            return 4
        elif category_name == 'Entity-Destination':
            return 5
        elif category_name == 'Component-Whole':
            return 6
        elif category_name == 'Member-Collection':
            return 7
        elif category_name == 'Message-Topic':
            return 8
        elif category_name == 'Other':
            return 9
        else:
            return -1

    @staticmethod
    def number_to_label(label_no):
        if label_no == 0:
            return 'Cause-Effect'
        elif label_no == 1:
            return 'Instrument-Agency'
        elif label_no == 2:
            return 'Product-Producer'
        elif label_no == 3:
            return 'Content-Container'
        elif label_no == 4:
            return 'Entity-Origin'
        elif label_no == 5:
            return 'Entity-Destination'
        elif label_no == 6:
            return 'Component-Whole'
        elif label_no == 7:
            return 'Member-Collection'
        elif label_no == 8:
            return 'Message-Topic'
        elif label_no == 9:
            return 'Other'
        else:
            return -1
