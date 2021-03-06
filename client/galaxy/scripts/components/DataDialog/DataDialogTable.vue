<template>
    <div>
        <b-table
            small
            hover
            :items="items"
            :fields="fields"
            :filter="filter"
            :per-page="perPage"
            :current-page="currentPage"
            @row-clicked="clicked"
            @filtered="filtered"
        >
            <template slot="label" slot-scope="data">
                <i v-if="data.item.isDataset" class="fa fa-file-o" /> <i v-else class="fa fa-folder" />
                {{ data.value ? data.value : "-" }}
            </template>
            <template slot="details" slot-scope="data">
                {{ data.value ? data.value : "-" }}
            </template>
            <template slot="time" slot-scope="data">
                {{ data.value ? data.value : "-" }}
            </template>
        </b-table>
        <div v-if="nItems === 0">
            <div v-if="filter">
                No search results found for: <b>{{ this.filter }}</b
                >.
            </div>
            <div v-else>No entries.</div>
        </div>
        <b-pagination v-if="nItems > perPage" v-model="currentPage" :per-page="perPage" :total-rows="nItems" />
    </div>
</template>

<script>
import Vue from "vue";
import BootstrapVue from "bootstrap-vue";

Vue.use(BootstrapVue);

export default {
    props: {
        items: {
            type: Array,
            required: true
        },
        filter: {
            type: String,
            default: null
        },
        multiple: {
            type: Boolean,
            default: false
        }
    },
    data() {
        return {
            currentPage: 1,
            fields: {
                label: {
                    sortable: true
                },
                details: {
                    sortable: true
                },
                time: {
                    sortable: true
                }
            },
            nItems: 0,
            perPage: 100
        };
    },
    watch: {
        items: {
            immediate: true,
            handler(items) {
                this.filtered(items);
            }
        }
    },
    methods: {
        /** Resets pagination when a filter/search word is entered **/
        filtered: function(items) {
            this.nItems = items.length;
            this.currentPage = 1;
        },
        /** Collects selected datasets in value array **/
        clicked: function(record) {
            this.$emit("clicked", record);
        }
    }
};
</script>
