#include <iostream>
#include <nlohmann/json.hpp>

#include "llama.h"
#include "common.h"
#include "chat.h"

using json = nlohmann::ordered_json;

static const std::string QWEN_TEMPLATE = R"({{-% if bos_token %}{{ bos_token }}{% endif %}
{{ system_prompt | default(\"\") }}
{% for message in messages %}
{% if message.role == \"assistant\" %}
{{ assistant_token }}{{ message.content }}
{% elif message.role == \"user\" %}
{{ user_token }}{{ message.content }}
{% elif message.role == \"tool\" %}
{{ tool_token }}{{ message.content }}
{% endif %}
{% endfor %}
{{ assistant_token }}
})";

int main() {
    // Minimal input JSON
    std::string json_str = R"({
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
        \"bos_token\": \"<s>\",
        \"eos_token\": \"</s>\",
        \"add_generation_prompt\": true
    })";
    json input = json::parse(json_str);

    auto tmpls = common_chat_templates_init(nullptr, QWEN_TEMPLATE, "<s>", "</s>");
    if (!tmpls) {
        std::cerr << "Failed to init chat template" << std::endl;
        return 1;
    }
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.messages = common_chat_msgs_parse_oaicompat(input["messages"]);
    inputs.add_generation_prompt = true;
    try {
        std::string prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt;
        if (prompt.empty()) {
            std::cerr << "Empty prompt result" << std::endl;
            return 1;
        }
        std::cout << "Prompt generated successfully" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
