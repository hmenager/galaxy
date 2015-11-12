

def json_wrap(inputs, input_values, job):
    for input in inputs.itervalues():
        input_name = input.name
        input_type = input.type
        value = input_values[input_name]
        if input_type == "repeat":
            repeat_job_value = []
            for d in input_values[input.name]:
                repeat_instance_job_value = {}
                json_wrap(input.inputs, d, repeat_instance_job_value)
                repeat_job_value.append(repeat_instance_job_value)
            job[input_name] = repeat_job_value
        if input_type == "conditional":
            values = input_values[input_name]
            current = values["__current_case__"]
            conditional_job_value = {}
            json_wrap(input.cases[current].inputs, values, conditional_job_value)
            job[input_name] = conditional_job_value
        if input_type == "section":
            values = input_values[input_name]
            section_job_value = {}
            json_wrap(input.inputs, values, section_job_value)
            job[input_name] = section_job_value
        elif input_type == "data" and input.multiple:
            raise NotImplementedError()
        elif input_type == "data":
            raise NotImplementedError()
        elif input_type == "data_collection" :
            raise NotImplementedError()
        elif input_type == "section" or input_type == "text":
            value = input_values[input_name]
            json_value = _cast_if_not_none(value, str)
            job[input_name] = json_value
        elif input_type == "float":
            value = input_values[input_name]
            json_value = _cast_if_not_none(value, float)
            job[input_name] = json_value
        elif input_type == "int":
            value = input_values[input_name]
            json_value = _cast_if_not_none(value, int)
            job[input_name] = json_value
        elif input_type == "boolean":
            value = input_values[input_name]
            json_value = _cast_if_not_none(value, bool)
            job[input_name] = json_value
        else:
            raise NotImplementedError()


def _cast_if_not_none(value, cast_to):
    if value is None:
        return value
    else:
        return cast_to(value)


__all__ = ['json_wrap']
