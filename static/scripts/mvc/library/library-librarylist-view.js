define(["layout/masthead","mvc/base-mvc","utils/utils","libs/toastr","mvc/library/library-model","mvc/library/library-libraryrow-view","libs/underscore"],function(a,b,c,d,e,f,g){var h=Backbone.View.extend({el:"#libraries_element",events:{"click .sort-libraries-link":"sort_clicked"},defaults:{page_count:null,show_page:null},initialize:function(a){this.options=g.defaults(this.options||{},a,this.defaults);var b=this;this.modal=null,this.collection=new e.Libraries,this.collection.fetch({success:function(){b.render()},error:function(a,b){d.error("undefined"!=typeof b.responseJSON?b.responseJSON.err_msg:"An error ocurred.")}})},render:function(a){this.options=g.extend(this.options,a),this.setElement("#libraries_element");var b=this.templateLibraryList(),c=null,d=null;if($(".tooltip").hide(),"undefined"!=typeof a&&(d="undefined"!=typeof a.models?a.models:null),null!==this.collection&&null===d)this.sortLibraries(),c=Galaxy.libraries.preferences.get("with_deleted")?this.collection.models:this.collection.where({deleted:!1});else if(null!==d)if(Galaxy.libraries.preferences.get("with_deleted"))c=d;else{var e=function(a){return a.get("deleted")===!1};c=g.filter(d,e)}else c=[];(null===this.options.show_page||this.options.show_page<1)&&(this.options.show_page=1),this.options.total_libraries_count=c.length;var f=Galaxy.libraries.preferences.get("library_page_size")*(this.options.show_page-1);this.options.page_count=Math.ceil(this.options.total_libraries_count/Galaxy.libraries.preferences.get("library_page_size")),this.options.total_libraries_count>0&&f<this.options.total_libraries_count?(c=c.slice(f,f+Galaxy.libraries.preferences.get("library_page_size")),this.options.libraries_shown=c.length,Galaxy.libraries.preferences.get("library_page_size")*this.options.show_page>this.options.total_libraries_count+Galaxy.libraries.preferences.get("library_page_size")&&(c=[]),this.$el.html(b({length:1,order:Galaxy.libraries.preferences.get("sort_order"),search_term:Galaxy.libraries.libraryToolbarView.options.search_term})),Galaxy.libraries.libraryToolbarView.renderPaginator(this.options),this.renderRows(c)):(this.$el.html(b({length:0,order:Galaxy.libraries.preferences.get("sort_order"),search_term:Galaxy.libraries.libraryToolbarView.options.search_term})),Galaxy.libraries.libraryToolbarView.renderPaginator(this.options)),$("#center [data-toggle]").tooltip(),$("#center").css("overflow","auto")},renderRows:function(a){for(var b=0;b<a.length;b++){var c=a[b];this.renderOne({library:c})}},renderOne:function(a){var b=a.library,c=new f.LibraryRowView(b);this.$el.find("#library_list_body").append(c.el)},sort_clicked:function(){Galaxy.libraries.preferences.set("asc"===Galaxy.libraries.preferences.get("sort_order")?{sort_order:"desc"}:{sort_order:"asc"}),this.render()},sortLibraries:function(){"name"===Galaxy.libraries.preferences.get("sort_by")&&("asc"===Galaxy.libraries.preferences.get("sort_order")?this.collection.sortByNameAsc():"desc"===Galaxy.libraries.preferences.get("sort_order")&&this.collection.sortByNameDesc())},redirectToHome:function(){window.location="../"},redirectToLogin:function(){window.location="/user/login"},searchLibraries:function(a){var b=$.trim(a);if(""!==b){var c=null;c=this.collection.search(a),this.options.searching=!0,this.render({models:c})}else this.options.searching=!1,this.render()},templateLibraryList:function(){return tmpl_array=[],tmpl_array.push('<div class="library_container table-responsive">'),tmpl_array.push("<% if(length === 0) { %>"),tmpl_array.push("<% if(search_term.length > 0) { %>"),tmpl_array.push("<div>There are no libraries matching your search. Try different keyword.</div>"),tmpl_array.push("<% } else{ %>"),tmpl_array.push('<div>There are no libraries visible to you here. If you expected some to show up please consult the <a href="https://wiki.galaxyproject.org/Admin/DataLibraries/LibrarySecurity" target="_blank">library security wikipage</a> or visit the <a href="https://biostar.usegalaxy.org/" target="_blank">Galaxy support site</a>.</div>'),tmpl_array.push("<% }%>"),tmpl_array.push("<% } else{ %>"),tmpl_array.push('<table class="grid table table-condensed">'),tmpl_array.push("   <thead>"),tmpl_array.push('     <th style="width:30%;"><a class="sort-libraries-link" title="Click to reverse order" href="#">name</a> <span title="Sorted alphabetically" class="fa fa-sort-alpha-<%- order %>"></span></th>'),tmpl_array.push('     <th style="width:22%;">description</th>'),tmpl_array.push('     <th style="width:22%;">synopsis</th> '),tmpl_array.push('     <th style="width:26%;"></th>'),tmpl_array.push("   </thead>"),tmpl_array.push('   <tbody id="library_list_body">'),tmpl_array.push("   </tbody>"),tmpl_array.push("</table>"),tmpl_array.push("<% }%>"),tmpl_array.push("</div>"),g.template(tmpl_array.join(""))}});return{LibraryListView:h}});
//# sourceMappingURL=../../../maps/mvc/library/library-librarylist-view.js.map