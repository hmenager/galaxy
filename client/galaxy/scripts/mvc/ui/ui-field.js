/** Renders the color picker used e.g. in the tool form **/
import Utils from "utils/utils";
import Ui from "mvc/ui/ui-misc";

/** Renders an input element used e.g. in the tool form */
export default Backbone.View.extend({
    initialize: function(options) {
        this.model =
            (options && options.model) ||
            new Backbone.Model({
                type: "text",
                placeholder: "",
                disabled: false,
                readonly: false,
                visible: true,
                cls: "",
                area: false,
                color: null,
                style: null
            }).set(options);
        this.tagName = this.model.get("area") ? "textarea" : "input";
        this.setElement($(`<${this.tagName}/>`));
        this.listenTo(this.model, "change", this.render, this);
        this.render();
    },
    events: {
        input: "_onchange"
    },
    value: function(new_val) {
        new_val !== undefined && this.model.set("value", typeof new_val === "string" ? new_val : "");
        return this.model.get("value");
    },
    render: function() {
        var self = this;
        this.$el
            .removeClass()
            .addClass(`ui-${this.tagName}`)
            .addClass(this.model.get("cls"))
            .addClass(this.model.get("style"))
            .addClass("awesomesaucefield")
            .attr("id", this.model.id)
            .attr("type", this.model.get("type"))
            .attr("placeholder", this.model.get("placeholder"))
            .css("color", this.model.get("color") || "")
            .css("border-color", this.model.get("color") || "");
        var datalist = this.model.get("datalist");
        if ($.isArray(datalist) && datalist.length > 0) {
            this.$el.autocomplete({
                source: function(request, response) {
                    response(self.model.get("datalist"));
                },
                change: function() {
                    self._onchange();
                }
            });
        }
        if (this.model.get("value") !== this.$el.val()) {
            this.$el.val(this.model.get("value"));
        }
        _.each(["readonly", "disabled"], attr_name => {
            self.model.get(attr_name) ? self.$el.attr(attr_name, true) : self.$el.removeAttr(attr_name);
        });
        this.$el[this.model.get("visible") ? "show" : "hide"]();
        return this;
    },
    _onchange: function() {
        this.value(this.$el.val());
        this.model.get("onchange") && this.model.get("onchange")(this.model.get("value"));
    }
});
